"""DeepSeek search and ordered, profile-scoped page extraction."""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from agent.secret_scope import UnscopedSecretError
from agent.web_search_provider import WebSearchProvider

from .client import DeepSeekError, is_available, request_json


def _url_error(url) -> str | None:
    """Reject fetch URLs outside the supported local URL contract."""
    if not isinstance(url, str) or len(url) > 8192:
        return "URL must be an http/https string of at most 8192 characters"
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
            return "URL must use http/https, have a host and contain no userinfo"
        parsed.port
    except ValueError:
        return "Invalid URL"
    return None


class DeepSeekWebSearchProvider(WebSearchProvider):
    """Hermes search and extraction backed by DeepSeek tools."""
    # Stable registry identity and picker label.
    name = "deepseek"
    display_name = "DeepSeek"

    def is_available(self):
        """Probe profile credentials without network access or exceptions."""
        return is_available()

    def supports_search(self):
        """Advertise the Hermes search capability."""
        return True

    def supports_extract(self):
        """Advertise the Hermes page-extraction capability."""
        return True

    def get_setup_schema(self):
        """Expose the per-profile API key prompt in the Hermes picker."""
        return {"name": "DeepSeek", "badge": "paid", "tag": "DeepSeek tools",
                "env_vars": [{"key": "DEEPSEEK_API_KEY", "prompt": "DeepSeek API key",
                              "url": "https://platform.deepseek.com/api_keys"}]}

    def search(self, query, limit=5):
        """Validate a query and map DeepSeek results into the Hermes search envelope."""
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Search query must be non-empty")
            query = query.strip()
            if len(query) > 128:
                raise ValueError("DeepSeek search query must be at most 128 characters")

            result = request_json("search", json={"query": query, "count": min(max(int(limit), 1), 50)})

            entries = []
            for index, item in enumerate(result["results"]):
                entry = {"title": item.get("title", ""), "url": item["url"],
                         "description": item.get("snippet") or item.get("summary", ""), "position": index + 1}
                entry.update({key: item[key] for key in ("summary", "date", "site_name", "authority", "rerank_score") if key in item})
                entries.append(entry)
            return {"success": True, "data": {"web": entries}}
        except UnscopedSecretError:
            raise
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def extract(self, urls, **kwargs):
        """Fetch URLs in input order with at most four scoped worker calls."""
        semaphore = asyncio.Semaphore(4)

        def fetch(url):
            """Map one fetch response or per-URL failure without hiding scoping bugs."""
            try:
                result = request_json("fetch", json={"url": url})
                # /fetch returns the page body as Markdown in ``content``.
                markdown = result["content"]
                if not isinstance(markdown, str):
                    raise DeepSeekError("DeepSeek fetch returned invalid content")
                return {"url": url, "title": result.get("title", ""), "content": markdown,
                        "raw_content": markdown, "metadata": {"source": "deepseek"}}
            except UnscopedSecretError:
                raise
            except (DeepSeekError, ValueError, TypeError, KeyError) as exc:
                return {"url": url, "title": "", "content": "", "raw_content": "", "error": str(exc)}

        async def one(url):
            """Validate one URL before acquiring a worker slot."""
            error = _url_error(url)
            if error:
                return {"url": url, "title": "", "content": "", "raw_content": "", "error": error}
            async with semaphore:
                # to_thread copies contextvars, including Hermes home and secret scope.
                return await asyncio.to_thread(fetch, url)

        return await asyncio.gather(*(one(url) for url in urls))
