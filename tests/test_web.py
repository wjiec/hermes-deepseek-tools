import json
import threading
import time

import httpx
import pytest


def test_registration(hermes):
    """Verify registration."""
    from agent import transcription_registry, tts_registry, web_search_registry
    from tools.registry import registry

    assert web_search_registry.get_provider("deepseek").supports_search()
    assert web_search_registry.get_provider("deepseek").supports_extract()
    assert tts_registry.get_provider("deepseek") is hermes.tts
    assert transcription_registry.get_provider("deepseek") is hermes.stt
    assert registry.get_entry("deepseek_ocr") is not None
    assert registry.get_toolset_for_tool("deepseek_ocr") == "deepseek"
    assert hermes.web.get_setup_schema()["env_vars"][0]["key"] == "DEEPSEEK_API_KEY"


def test_search_real_tool(hermes):
    """Verify search real tool."""
    from tools.web_tools import web_search_tool

    hermes.respond(lambda r: httpx.Response(200, json={"results": [
        {"title": "One", "url": "https://example.com/one", "snippet": "snippet", "summary": "summary",
         "date": "2026-11-01", "site_name": "Example", "authority": 0.9, "rerank_score": 0.8},
        {"title": "Two", "url": "https://example.com/two", "summary": "fallback"},
    ]}))
    result = json.loads(web_search_tool("  query  ", limit=5))
    assert result["success"]
    first, second = result["data"]["web"]
    assert first == {"title": "One", "url": "https://example.com/one", "description": "snippet", "position": 1,
                     "summary": "summary", "date": "2026-11-01", "site_name": "Example", "authority": 0.9, "rerank_score": 0.8}
    assert second["description"] == "fallback" and second["position"] == 2
    request, = hermes.requests
    assert request.method == "POST" and request.url.path == "/v1/tools/search"
    # Hermes buckets requested result counts even when its memo is disabled.
    assert json.loads(request.content) == {"query": "query", "count": 10}
    assert request.headers["Authorization"] == "Bearer test-key"
    assert "hermes-deepseek-tools" in request.headers["User-Agent"]


@pytest.mark.parametrize("status,expected,count", [(401, "rejected the API key for profile 'default'", 1),
                                                   (402, "balance is insufficient", 1), (429, "upstream failure", 2),
                                                   (500, "upstream failure", 2)])
def test_search_errors(hermes, status, expected, count):
    """Verify search errors."""
    from tools.web_tools import web_search_tool

    hermes.respond(lambda r: httpx.Response(status, json={"error": {"message": "upstream failure"}}))
    result = json.loads(web_search_tool("query"))
    assert not result["success"] and expected in result["error"]
    assert len(hermes.requests) == count


@pytest.mark.parametrize("query", ["x" * 129, "   ", None])
def test_search_local_rejection(hermes, query):
    """Verify search local rejection."""
    from tools.web_tools import web_search_tool

    assert not json.loads(web_search_tool(query))["success"]
    assert hermes.requests == []


def test_search_count_clamped(hermes):
    """Verify search count clamped."""
    hermes.respond(lambda r: httpx.Response(200, json={"results": []}))
    hermes.web.search("query", 500)
    hermes.web.search("query", -1)
    assert [json.loads(r.content)["count"] for r in hermes.requests] == [50, 1]


async def test_extract_real_tool(hermes):
    """Verify extract real tool."""
    from tools.web_tools import web_extract_tool

    urls = ["https://example.com/one", "https://example.com/missing", "https://example.com/three"]

    def respond(request):
        """Supply the upstream outcome and request timing required by this test."""
        url = json.loads(request.content)["url"]
        if url.endswith("missing"):
            return httpx.Response(404)
        return httpx.Response(200, json={"title": url.rsplit("/", 1)[-1], "content": "# " + url})

    hermes.respond(respond)
    result = json.loads(await web_extract_tool(urls))
    entries = result["results"]
    assert [entry["url"] for entry in entries] == urls
    assert entries[0]["content"] == "# " + urls[0]
    assert "page could not be fetched" in entries[1]["error"]
    assert entries[1]["content"] == "" and entries[2]["error"] is None
    assert len(hermes.requests) == 3
    assert all(r.url.path == "/v1/tools/fetch" and r.method == "POST" for r in hermes.requests)


@pytest.mark.parametrize("url", ["ftp://example.com/file", "https:///x", "https://user:pass@example.com", "https://example.com/" + "x" * 8192], ids=["scheme", "host", "userinfo", "length"])
async def test_extract_local_url_validation(hermes, url):
    """Verify extract local url validation."""
    result = await hermes.web.extract([url], format="text", include_raw=False, max_chars=10)
    assert result[0]["error"] and result[0]["content"] == ""
    assert hermes.requests == []


async def test_extract_invalid_real_tool(hermes):
    """Verify extract invalid real tool."""
    from tools.web_tools import web_extract_tool

    result = json.loads(await web_extract_tool(["data:text/plain,bad"]))
    assert result["results"][0]["error"]
    assert hermes.requests == []


async def test_extract_missing_key_real_tool(hermes, monkeypatch):
    """Verify extract missing key real tool."""
    from tools.web_tools import web_extract_tool

    monkeypatch.delenv("DEEPSEEK_API_KEY")
    result = json.loads(await web_extract_tool(["https://example.com/page"]))
    assert "not set for profile 'default'" in result["results"][0]["error"]
    assert hermes.requests == []


async def test_extract_concurrency_and_order(hermes):
    """Verify extract concurrency and order."""
    lock = threading.Lock()
    active = 0
    peak = 0

    def respond(request):
        """Supply the upstream outcome and request timing required by this test."""
        nonlocal active, peak
        url = json.loads(request.content)["url"]
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return httpx.Response(200, json={"title": "title", "content": url})

    hermes.respond(respond)
    urls = [f"https://example.com/{index}" for index in range(10)]
    result = await hermes.web.extract(urls)
    assert [entry["url"] for entry in result] == urls
    assert [entry["raw_content"] for entry in result] == urls
    assert all(entry["metadata"] == {"source": "deepseek"} for entry in result)
    assert 1 < peak <= 4
