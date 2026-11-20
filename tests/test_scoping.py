import asyncio
import json

import httpx
import pytest


def _media(home):
    """Create a profile-owned OCR attachment for credential-routing checks."""
    path = home / "cache/documents/upload.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    return path


def _call_tools(hermes, home):
    """Exercise the four real Hermes entrypoints in the current profile scope."""
    from tools.registry import registry
    from tools.transcription_tools import transcribe_audio
    from tools.tts_tool import text_to_speech_tool
    from tools.web_tools import web_search_tool

    return [json.loads(web_search_tool("same query")),
            json.loads(text_to_speech_tool("hello", str(home / "reply.mp3"))),
            transcribe_audio(str(hermes.audio_dir / "voice.ogg")),
            json.loads(registry.dispatch("deepseek_ocr", {"path": str(_media(home))}))]


def test_multiplex_credentials_at_real_entries(hermes, wav_bytes, monkeypatch):
    """Verify multiplex credentials at real entries."""
    from agent.runtime_scope import bind_profile_scope
    from agent.secret_scope import invalidate_env_file_cache, set_multiplex_active

    default = hermes.home
    alice = default / "profiles/alice"
    hermes.configure(alice)
    (default / ".env").write_text("DEEPSEEK_API_KEY=key-default\n")
    (alice / ".env").write_text("DEEPSEEK_API_KEY=key-alice\n")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key-default")

    def respond(request):
        """Supply the upstream outcome and request timing required by this test."""
        endpoint = request.url.path.rsplit("/", 1)[-1]
        if endpoint == "tts":
            return httpx.Response(200, content=wav_bytes)
        bodies = {"search": {"results": []}, "asr": {"text": "speech"}, "ocr": {"pages": [{"text": "document"}]},
                  "fetch": {"title": "page", "content": "content"}}
        return httpx.Response(200, json=bodies[endpoint])

    hermes.respond(respond)
    set_multiplex_active(True)
    for home, key in [(default, "key-default"), (alice, "key-alice")]:
        with bind_profile_scope(home):
            loaded = hermes.install(home) if home == alice else hermes
            start = len(hermes.requests)
            results = _call_tools(hermes, home)
            assert all(result["success"] for result in results), results
            # Also exercise context propagation into concurrent fetch threads.
            fetched = asyncio.run(loaded.web.extract(["https://example.com/1", "https://example.com/2"]))
            assert all(entry["content"] == "content" for entry in fetched)
            sent = hermes.requests[start:]
            assert {r.url.path for r in sent} == {"/v1/tools/search", "/v1/tools/tts", "/v1/tools/asr", "/v1/tools/ocr", "/v1/tools/fetch"}
            assert len(sent) == 6
            assert {r.headers["Authorization"] for r in sent} == {f"Bearer {key}"}

    (alice / ".env").write_text("")
    invalidate_env_file_cache(alice / ".env")
    with bind_profile_scope(alice):
        start = len(hermes.requests)
        results = _call_tools(hermes, alice)
        assert all(not result["success"] for result in results), results
        assert len(hermes.requests) == start
        for index in (0, 1, 3):
            assert "not set for profile 'alice'" in results[index]["error"]
        # Hermes STT gates on availability, so the precise credential failure is
        # visible at the provider boundary rather than the transcribe_audio gate.
        direct = loaded.stt.transcribe(str(hermes.audio_dir / "voice.ogg"))
        assert "not set for profile 'alice'" in direct["error"]
        assert len(hermes.requests) == start
        assert not loaded.web.is_available() and not loaded.tts.is_available() and not loaded.stt.is_available()
        assert not loaded.ocr.check_ocr_available()
    set_multiplex_active(False)


def test_multiplex_no_scope_fails_closed(hermes, tmp_path):
    """Verify multiplex no scope fails closed."""
    from agent.secret_scope import UnscopedSecretError, set_multiplex_active
    from tools.registry import registry
    from tools.web_tools import web_search_tool

    set_multiplex_active(True)
    with pytest.raises(UnscopedSecretError, match="no profile secret scope"):
        hermes.web.search("query")
    with pytest.raises(UnscopedSecretError, match="no profile secret scope"):
        hermes.tts.synthesize("hello", str(tmp_path / "reply.mp3"))
    result = hermes.stt.transcribe(str(hermes.audio_dir / "voice.ogg"))
    assert not result["success"] and "internal profile-scoping bug" in result["error"]
    result = json.loads(registry.dispatch("deepseek_ocr", {"path": str(_media(hermes.home))}))
    assert "internal profile-scoping bug" in result["error"]
    assert "internal profile-scoping bug" in json.loads(web_search_tool("query"))["error"]
    assert not hermes.web.is_available() and not hermes.tts.is_available() and not hermes.stt.is_available()
    assert not hermes.ocr.check_ocr_available()
    assert hermes.requests == []
    set_multiplex_active(False)


def test_missing_default_key_error(hermes, monkeypatch):
    """Verify missing default key error."""
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    result = hermes.web.search("query")
    assert result == {"success": False, "error": f"DEEPSEEK_API_KEY is not set for profile 'default' (add it to {hermes.home}/.env)"}
    assert hermes.requests == []
