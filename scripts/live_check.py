"""Live acceptance check against the real DeepSeek API, run on a Hermes host.

Loads the installed ``deepseek-tools`` plugin through Hermes' own discovery inside each
profile's scope with multiplexing switched on (the code path a multiplexed gateway uses),
then exercises Search, Fetch, TTS, ASR and OCR once per profile. It changes no
configuration: providers are called directly, so profiles keep their configured backends.

Cost per profile: 1 search + 1 fetch + 1 short TTS + 2 short ASR + 1 one-page OCR.
Nothing secret is printed: keys appear only as an 8-character SHA-256 fingerprint.

Usage (any python3; the script re-launches itself inside Hermes' own runtime, located with
``hermes --print-runtime-command``, when Hermes is not importable):

    python3 scripts/live_check.py --profile default --profile <name> --json live-check.json
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

FETCH_URL = "https://api-docs.deepseek.com/zh-cn/guides/search/"
TTS_TEXT = "Hermes DeepSeek tools acceptance test. 插件验收测试。"
OCR_MARKER = "HERMES DEEPSEEK OCR CHECK"


def _pdf_with_text(text: str) -> bytes:
    """Smallest valid one-page PDF that draws *text* in Helvetica (ASCII only)."""
    stream = f"BT /F1 28 Tf 72 700 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def _probe(path: Path) -> str:
    probe = shutil.which("ffprobe")
    if not probe:
        return "ffprobe missing"
    result = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration:stream=codec_name,sample_rate,channels",
                             "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return " ".join(result.stdout.split())


def _step(results: list, name: str, fn):
    started = time.monotonic()
    try:
        ok, detail = fn()
    except Exception as exc:  # report and continue with the next capability
        ok, detail = False, f"{type(exc).__name__}: {exc}"
        if "--debug" in sys.argv:
            traceback.print_exc()
    results.append({"check": name, "ok": bool(ok), "seconds": round(time.monotonic() - started, 2), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<14} {results[-1]['seconds']:>6}s  {detail}", flush=True)


def check_profile(home: Path, work: Path) -> dict:
    from agent.runtime_scope import bind_profile_scope
    from agent.secret_scope import get_secret
    from agent import transcription_registry, tts_registry, web_search_registry
    from hermes_cli import plugins
    from tools.registry import registry

    results: list = []
    with bind_profile_scope(home):
        key = get_secret("DEEPSEEK_API_KEY") or ""
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8] if key else "(none)"
        plugins.discover_plugins(force=True)
        info = plugins.get_plugin_manager()._plugins.get("deepseek-tools")
        print(f"\nprofile home: {home}\n  key fingerprint: {fingerprint}   plugin enabled: {bool(info and info.enabled)}")
        if not (info and info.enabled):
            results.append({"check": "plugin", "ok": False, "detail": getattr(info, "error", None) or "not installed/enabled"})
            return {"home": str(home), "key_fingerprint": fingerprint, "results": results}

        web = web_search_registry.get_provider("deepseek")
        tts = tts_registry.get_provider("deepseek")
        stt = transcription_registry.get_provider("deepseek")

        def search():
            out = web.search("DeepSeek API documentation", 3)
            rows = (out.get("data") or {}).get("web") or []
            return out.get("success") and rows, f"{len(rows)} results; first: {rows[0]['url'] if rows else out.get('error')}"

        def fetch():
            row = asyncio.run(web.extract([FETCH_URL]))[0]
            return not row.get("error") and len(row.get("content", "")) > 200, \
                row.get("error") or f"title={row.get('title')!r} content={len(row['content'])} chars"

        wav = work / "tts.wav"

        def speak():
            path = Path(tts.synthesize(TTS_TEXT, str(work / "tts.mp3")))
            path.replace(wav)
            return wav.read_bytes()[:4] == b"RIFF", f"{wav.stat().st_size} bytes [{_probe(wav)}]"

        def transcribe_wav():
            if not wav.exists():
                return False, "skipped: TTS produced no audio"
            out = stt.transcribe(str(wav))
            return out.get("success") and out.get("transcript"), out.get("transcript") or out.get("error")

        def transcribe_voice_note():
            ffmpeg = shutil.which("ffmpeg")
            if not wav.exists():
                return False, "skipped: TTS produced no audio"
            if not ffmpeg:
                return False, "ffmpeg missing (needed to build an Ogg Opus voice note)"
            ogg = work / "voice.ogg"
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "48000", "-c:a", "libopus",
                            "-b:a", "24k", str(ogg)], check=True)
            out = stt.transcribe(str(ogg))
            return out.get("success") and out.get("transcript"), out.get("transcript") or out.get("error")

        def ocr():
            # Same place the gateway caches user documents, so the multiplex path policy applies.
            target = home / "cache" / "documents" / f"deepseek-live-check-{int(time.time())}.pdf"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_pdf_with_text(OCR_MARKER))
            try:
                out = json.loads(registry.dispatch("deepseek_ocr", {"path": str(target)}))
            finally:
                target.unlink(missing_ok=True)
            text = out.get("text", "")
            return out.get("success") and OCR_MARKER.split()[0] in text.upper(), \
                out.get("error") or f"pages={out.get('page_count')} text={' '.join(text.split())[:80]!r}"

        def ocr_refuses_outside_cache():
            outside = work / "outside.pdf"
            outside.write_bytes(_pdf_with_text(OCR_MARKER))
            out = json.loads(registry.dispatch("deepseek_ocr", {"path": str(outside)}))
            return not out.get("success") and "media cache" in str(out.get("error")), str(out.get("error"))

        _step(results, "search", search)
        _step(results, "fetch", fetch)
        _step(results, "tts", speak)
        _step(results, "asr (wav)", transcribe_wav)
        _step(results, "asr (ogg opus)", transcribe_voice_note)
        _step(results, "ocr", ocr)
        _step(results, "ocr scope", ocr_refuses_outside_cache)
    return {"home": str(home), "key_fingerprint": fingerprint, "results": results}


def _ensure_hermes_runtime() -> None:
    """Re-exec under the interpreter and source tree the installed ``hermes`` launcher uses."""
    try:
        import hermes_constants  # noqa: F401
        return
    except ImportError:
        pass
    if os.environ.get("_DEEPSEEK_LIVE_CHECK_REEXEC"):
        sys.exit("live_check: Hermes is still not importable inside its runtime")
    hermes = shutil.which("hermes")
    if not hermes:
        sys.exit("live_check: `hermes` is not on PATH")
    env = dict(os.environ, _DEEPSEEK_LIVE_CHECK_REEXEC="1")
    probe = subprocess.run([hermes, "--print-runtime-command"], capture_output=True, text=True)
    if probe.returncode != 0:
        # pip/uv installs: the launcher is a console script whose shebang is Hermes' interpreter.
        with open(hermes, "rb") as launcher:
            shebang = launcher.readline().decode(errors="replace")
        if not shebang.startswith("#!") or "python" not in shebang:
            sys.exit("live_check: cannot locate the Python interpreter that runs Hermes")
        python = shebang[2:].strip().split()[0]
        os.execve(python, [python, os.path.abspath(__file__), *sys.argv[1:]], env)
    argv = json.loads(probe.stdout)
    match = re.search(r"sys\.path\.insert\(0, '([^']+)'\)", argv[-1])
    if not match:
        sys.exit("live_check: could not read the Hermes source root from --print-runtime-command")
    prelude = ("import os, sys, runpy; os.environ.pop('PYTHONHOME', None); os.environ.pop('PYTHONPATH', None); "
               f"sys.path.insert(0, {match.group(1)!r}); sys._hermes_pin_default_home = True; import hermes_bootstrap; "
               "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    os.execve(argv[0], [argv[0], "-I", "-c", prelude, os.path.abspath(__file__), *sys.argv[1:]], env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", action="append", required=True, help="profile name ('default' for the root home)")
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    parser.add_argument("--debug", action="store_true", help="print tracebacks")
    args = parser.parse_args()
    _ensure_hermes_runtime()

    from agent.secret_scope import set_multiplex_active
    from hermes_cli.version_info import hermes_user_agent
    from hermes_constants import get_default_hermes_root

    root = get_default_hermes_root()
    set_multiplex_active(True)  # fail closed exactly like a multiplexed gateway
    report = {"hermes": hermes_user_agent("hermes"), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "profiles": {}}
    print(f"{report['hermes']}  root={root}")
    with tempfile.TemporaryDirectory(prefix="deepseek-live-") as work:
        for name in args.profile:
            home = root if name == "default" else root / "profiles" / name
            (Path(work) / name).mkdir()
            report["profiles"][name] = check_profile(home, Path(work) / name)

    prints = [p["key_fingerprint"] for p in report["profiles"].values()]
    distinct = len(set(prints)) == len(prints)
    passed = all(r["ok"] for p in report["profiles"].values() for r in p["results"])
    report.update(passed=passed, distinct_keys=distinct)
    print(f"\nkeys distinct per profile: {distinct}   overall: {'PASS' if passed else 'FAIL'}")
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
