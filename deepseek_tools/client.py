"""Profile-scoped DeepSeek HTTP boundary.

Tests may inject ``httpx.MockTransport`` via ``_transport`` and replace ``_sleep``.
Clients are short-lived; credentials are resolved for every HTTP attempt.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager

import httpx
from agent.retry_utils import parse_retry_after_seconds
from agent.secret_scope import get_secret
from hermes_cli.version_info import hermes_user_agent
from hermes_constants import get_hermes_home, profile_name_for_home

# All tool requests stay on this credentialed origin.
BASE_URL = "https://api.deepseek.com/v1/tools"
# Test-only injection points; production creates a fresh transport per client.
_transport: httpx.BaseTransport | None = None
_sleep = time.sleep
# The outbound boundary logs only endpoint, status and elapsed seconds.
logger = logging.getLogger(__name__)
# Only these failures on these endpoints permit one additional attempt.
_RETRYABLE = {408, 429, 500, 503, 504}
_IDEMPOTENT = {"search", "fetch", "asr"}


class DeepSeekError(Exception):
    """A safe upstream failure with status and retry metadata."""
    def __init__(self, message, status=None, retryable=False):
        """Carry a safe message, HTTP status and retry classification."""
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def api_key() -> str:
    """Read the active profile key or report its profile-specific setup path."""
    key = (get_secret("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        home = get_hermes_home()
        profile = profile_name_for_home(home) or home.name
        raise DeepSeekError(f"DEEPSEEK_API_KEY is not set for profile '{profile}' (add it to {home}/.env)")
    return key


def is_available() -> bool:
    """Probe profile credentials without network access or exceptions."""
    try:
        return bool(api_key())
    except Exception:
        return False


def _redact(message: str, key: str, payload: dict) -> str:
    """Remove credential and complete request-body echoes from upstream diagnostics."""
    values = [key]
    for body in (payload.get("json", {}), payload.get("data", {})):
        for value in body.values():
            if isinstance(value, str) and value:
                values.append(value)

    # An upstream diagnostic can echo an upload as well as a JSON field. Files
    # are seekable local inputs; restore their position after inspecting them.
    for filename, content, mime in payload.get("files", {}).values():
        if not isinstance(content, bytes):
            position = content.tell()
            content.seek(0)
            value = content.read()
            content.seek(position)
        else:
            value = content
        values.append(value.decode("utf-8", errors="replace"))

    # Replace complete bodies first: replacing a key embedded in an echoed body
    # first would prevent matching the rest of that body.
    for value in sorted(values, key=len, reverse=True):
        if value:
            message = message.replace(value, "[redacted]")
    return message


def _response_error(response: httpx.Response, endpoint: str, key: str, payload: dict) -> DeepSeekError:
    """Translate an upstream HTTP failure into a safe DeepSeek error."""
    status = response.status_code
    message = None
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = _redact(error["message"], key, payload)
    except (ValueError, UnicodeError):
        pass

    profile = profile_name_for_home(get_hermes_home()) or get_hermes_home().name
    fixed = {
        401: f"DeepSeek rejected the API key for profile '{profile}'",
        402: "DeepSeek account balance is insufficient",
        413: "DeepSeek request is too large",
    }
    if endpoint == "fetch":
        fixed.update({403: "blocked by DeepSeek's fetch security policy",
                      404: "page could not be fetched (unavailable, blocked or no extractable content)"})
    message = fixed.get(status) or message or ("bad request" if status == 400 else f"DeepSeek HTTP {status}")
    return DeepSeekError(message, status=status, retryable=status in _RETRYABLE)


def _http_client(key: str, timeout) -> httpx.Client:
    """Create a redirect-disabled client for one credentialed attempt."""
    return httpx.Client(
        headers={"Authorization": f"Bearer {key}", "User-Agent": hermes_user_agent("hermes-deepseek-tools")},
        transport=_transport, follow_redirects=False, timeout=timeout,
    )


def request_json(endpoint: str, *, timeout=130, **payload) -> dict:
    """Post a tool request, retrying only the three idempotent endpoints once."""
    attempts = 2 if endpoint in _IDEMPOTENT else 1
    for attempt in range(attempts):
        key = api_key()
        started = time.monotonic()
        status = None
        delay = 1.0
        try:
            with _http_client(key, timeout) as client:
                response = client.post(f"{BASE_URL}/{endpoint}", **payload)

            status = response.status_code
            if not response.is_success:
                retry_after = parse_retry_after_seconds(response.headers)
                delay = min(retry_after, 5) if retry_after is not None else 1
                raise _response_error(response, endpoint, key, payload)

            try:
                result = json.loads(response.content)
            except (ValueError, UnicodeError):
                raise DeepSeekError("DeepSeek returned invalid JSON") from None
            if not isinstance(result, dict):
                raise DeepSeekError("DeepSeek returned an invalid response object")
            if "error" in result:
                error = result["error"]
                message = error.get("message", "DeepSeek request failed") if isinstance(error, dict) else str(error)
                raise DeepSeekError(_redact(str(message), key, payload))
            return result
        except httpx.TransportError:
            error = DeepSeekError("DeepSeek transport error", retryable=True)
        except DeepSeekError as exc:
            error = exc
        finally:
            logger.info("/%s status=%s elapsed=%.3fs", endpoint, status, time.monotonic() - started)
        if not error.retryable or attempt + 1 == attempts:
            raise error from None
        _sleep(delay)
    raise AssertionError("request loop must return or raise")


@contextmanager
def stream_tts(body: dict):
    """One billed synthesis attempt; never retry a partial or failed response."""
    key = api_key()
    started = time.monotonic()
    status = None
    try:
        with _http_client(key, 130) as client:
            with client.stream("POST", f"{BASE_URL}/tts", json=body) as response:
                status = response.status_code
                if not response.is_success:
                    response.read()
                    raise _response_error(response, "tts", key, {"json": body})
                yield response
    except httpx.TransportError:
        raise DeepSeekError("DeepSeek TTS stream closed prematurely or failed") from None
    finally:
        logger.info("/tts status=%s elapsed=%.3fs", status, time.monotonic() - started)
