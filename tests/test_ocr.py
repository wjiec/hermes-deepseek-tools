import json

import httpx
import pytest

from test_audio import multipart


@pytest.fixture
def pdf(hermes):
    """Create a one-page PDF with correct object offsets in the profile cache."""
    path = hermes.home / "cache/documents/attachment.pdf"
    path.parent.mkdir(parents=True)
    # One empty page, with a valid xref table; no PDF library is required.
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] >>"]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(b"xref\n0 4\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(data)
    return path


def dispatch(path, **args):
    """Execute OCR through the real Hermes tool registry."""
    from tools.registry import registry

    return json.loads(registry.dispatch("deepseek_ocr", {"path": str(path), **args}, task_id="offline-test"))


def test_ocr_real_registry(hermes, pdf):
    """Verify ocr real registry."""
    body = {"pages": [{"index": 1, "text": "First page", "finish_reason": "stop"},
                       {"index": 2, "text": "Second page", "finish_reason": "length"}]}
    hermes.respond(lambda r: httpx.Response(200, content=b"\n\n" + json.dumps(body).encode()))
    result = dispatch(pdf.as_uri(), purpose="rich_layout_recognition")
    assert result == {"success": True, "page_count": 2, "purpose": "rich_layout_recognition",
                      "text": "[page 1]\nFirst page\n\n---\n\n[page 2]\nSecond page\n[truncated]",
                      "pages": [{"index": 1, "text": "First page", "truncated": False},
                                {"index": 2, "text": "Second page", "truncated": True}]}
    request, = hermes.requests
    assert request.method == "POST" and request.url.path == "/v1/tools/ocr"
    parts = multipart(request)
    assert set(parts) == {"file", "purpose"}
    assert parts["file"].get_filename() == pdf.name
    assert parts["file"].get_payload(decode=True) == pdf.read_bytes()
    assert parts["purpose"].get_payload(decode=True) == b"rich_layout_recognition"
    assert request.extensions["timeout"]["connect"] == 10
    assert request.extensions["timeout"]["read"] == 190
    assert request.extensions["timeout"]["write"] == 130


def test_ocr_200_error(hermes, pdf):
    """Verify ocr 200 error."""
    hermes.respond(lambda r: httpx.Response(200, content=b'\n\n{"error":{"message":"could not read document"}}'))
    result = dispatch(pdf)
    assert not result["success"] and "could not read document" in result["error"]
    assert len(hermes.requests) == 1


@pytest.mark.parametrize("case", ["http", "data", "secret", "symlink_secret", "text", "missing", "directory", "purpose", "oversized", "zip"])
def test_ocr_local_refusals(hermes, pdf, case):
    """Verify ocr local refusals."""
    path = pdf
    kwargs = {}
    if case == "http":
        path = "http://example.com/attachment.pdf"
    elif case == "data":
        path = "data:application/pdf;base64,JVBERg=="
    elif case in {"secret", "symlink_secret"}:
        secret = hermes.home / ".env"
        secret.write_bytes(pdf.read_bytes())
        path = secret
        if case == "symlink_secret":
            path = pdf.parent / "linked.pdf"
            path.symlink_to(secret)
    elif case == "text":
        path = pdf.parent / "plain.txt"
        path.write_text("ordinary text")
    elif case == "missing":
        path = pdf.parent / "missing.pdf"
    elif case == "directory":
        path = pdf.parent
    elif case == "purpose":
        kwargs["purpose"] = "unknown"
    elif case == "oversized":
        path = pdf.parent / "oversized.pdf"
        with path.open("wb") as target:
            target.write(b"%PDF")
            target.truncate(100 * 1024 * 1024 + 1)
    elif case == "zip":
        path = pdf.parent / "archive.zip"
        path.write_bytes(b"PK\x03\x04" + b"\0" * 20)
    result = dispatch(path, **kwargs)
    assert "error" in result and not result.get("success", False)
    assert hermes.requests == []


def test_ocr_multiplex_location_policy(hermes, pdf, tmp_path):
    """Verify ocr multiplex location policy."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(pdf.read_bytes())
    set_multiplex_active(True)
    token = set_secret_scope({"DEEPSEEK_API_KEY": "scoped"}, profile_home=str(hermes.home))
    hermes.respond(lambda r: httpx.Response(200, json={"pages": [{"text": "recognized"}]}))
    try:
        result = dispatch(outside)
        assert "media cache" in result["error"]
        assert hermes.requests == []
        link = pdf.parent / "outside-link.pdf"
        link.symlink_to(outside)
        assert "media cache" in dispatch(link)["error"]
        assert hermes.requests == []
        result = dispatch(pdf)
        assert result["success"] and result["text"] == "recognized"
        assert len(hermes.requests) == 1
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_ocr_nonlocal_terminal(hermes, pdf, tmp_path, monkeypatch):
    """Verify ocr nonlocal terminal."""
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(pdf.read_bytes())
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    assert "media cache" in dispatch(outside)["error"]
    assert hermes.requests == []
    hermes.respond(lambda r: httpx.Response(200, json={"pages": [{"text": "local cache"}]}))
    assert dispatch(pdf)["success"]


def test_ocr_local_terminal_outside_allowed(hermes, pdf, tmp_path):
    """Verify ocr local terminal outside allowed."""
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(pdf.read_bytes())
    hermes.respond(lambda r: httpx.Response(200, json={"pages": [{"text": "outside allowed"}]}))
    assert dispatch(outside)["success"]
