import logging

import httpx
import pytest


def test_retry_after_and_key_resolved_each_attempt(hermes, monkeypatch):
    """Verify retry after and key resolved each attempt."""
    keys = iter(["first-key", "second-key"])
    monkeypatch.setattr(hermes.client, "get_secret", lambda name: next(keys))
    hermes.respond(lambda r: httpx.Response(503, headers={"Retry-After": "9"}) if len(hermes.requests) == 1 else httpx.Response(200, json={"results": []}))
    assert hermes.web.search("query")["success"]
    assert hermes.sleeps == [5]
    assert [r.headers["Authorization"] for r in hermes.requests] == ["Bearer first-key", "Bearer second-key"]


@pytest.mark.parametrize("endpoint", ["search", "fetch", "asr"])
def test_transport_retry(hermes, endpoint):
    """Verify transport retry."""
    def respond(request):
        """Supply the upstream outcome and request timing required by this test."""
        if len(hermes.requests) == 1:
            raise httpx.ConnectError("test-key in unsafe transport exception")
        return httpx.Response(200, json={"ok": True})

    hermes.respond(respond)
    assert hermes.client.request_json(endpoint, json={}) == {"ok": True}
    assert len(hermes.requests) == 2 and hermes.sleeps == [1]


@pytest.mark.parametrize("endpoint", ["tts", "ocr"])
@pytest.mark.parametrize("failure", ["status", "transport"])
def test_billed_calls_never_retry(hermes, endpoint, failure):
    """Verify billed calls never retry."""
    def respond(request):
        """Supply the upstream outcome and request timing required by this test."""
        if failure == "transport":
            raise httpx.ConnectError("test-key unsafe message")
        return httpx.Response(503, headers={"Retry-After": "3"})

    hermes.respond(respond)
    with pytest.raises(hermes.client.DeepSeekError):
        if endpoint == "tts":
            with hermes.client.stream_tts({"text": "hello", "voice_id": "stella", "format": "wav"}):
                pass
        else:
            hermes.client.request_json(endpoint, data={"purpose": "text_recognition"})
    assert len(hermes.requests) == 1 and hermes.sleeps == []


@pytest.mark.parametrize("status,endpoint,expected", [(400, "search", "bad request"),
                                                      (403, "fetch", "fetch security policy"),
                                                      (404, "fetch", "page could not be fetched"),
                                                      (413, "asr", "too large")])
def test_status_mapping_nonjson(hermes, status, endpoint, expected):
    """Verify status mapping nonjson."""
    hermes.respond(lambda r: httpx.Response(status, content=b"html error"))
    with pytest.raises(hermes.client.DeepSeekError, match=expected) as info:
        hermes.client.request_json(endpoint, json={})
    assert info.value.status == status and not info.value.retryable


def test_key_and_body_absent_from_errors_and_logs(hermes, caplog):
    """Verify key and body absent from errors and logs."""
    caplog.set_level(logging.INFO)
    hermes.respond(lambda r: httpx.Response(400, json={"error": {"message": "test-key rejected private request text"}}))
    with pytest.raises(hermes.client.DeepSeekError) as info:
        hermes.client.request_json("search", json={"query": "private request text"})
    assert "test-key" not in str(info.value)
    assert "private request text" not in str(info.value)
    assert "test-key" not in caplog.text and "private request text" not in caplog.text
    assert "/search status=400 elapsed=" in caplog.text


def test_transport_error_is_sanitized(hermes, caplog):
    """Verify transport error is sanitized."""
    def respond(request):
        """Supply the upstream outcome and request timing required by this test."""
        raise httpx.ReadError("test-key and sensitive body")

    caplog.set_level(logging.INFO)
    hermes.respond(respond)
    with pytest.raises(hermes.client.DeepSeekError) as info:
        hermes.client.request_json("search", json={"query": "sensitive body"})
    assert "test-key" not in str(info.value) + caplog.text
    assert "sensitive body" not in str(info.value) + caplog.text
    assert len(hermes.requests) == 2


@pytest.mark.parametrize("upload_type", ["bytes", "file"])
def test_upload_body_echo_is_redacted(hermes, tmp_path, caplog, upload_type):
    """Verify upload body echo is redacted."""
    content = b"%PDF-1.4\ntest-key\nPRIVATE DOCUMENT BODY\n%%EOF\n"
    path = tmp_path / "upload.pdf"
    path.write_bytes(content)
    caplog.set_level(logging.INFO)
    hermes.respond(lambda r: httpx.Response(400, json={"error": {"message": content.decode()}}))
    with path.open("rb") as source:
        with pytest.raises(hermes.client.DeepSeekError) as info:
            hermes.client.request_json("ocr", files={"file": (path.name, content if upload_type == "bytes" else source, "application/pdf")},
                                       data={"purpose": "text_recognition"})
    assert "PRIVATE DOCUMENT BODY" not in str(info.value) + caplog.text
    assert len(hermes.requests) == 1
