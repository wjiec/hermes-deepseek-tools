"""Real Hermes discovery, isolated homes, and an HTTP-only offline boundary."""
from __future__ import annotations

import importlib
import json
import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

# Hermes modules may resolve their home at import time, before fixtures run.
_session_home = tempfile.TemporaryDirectory(prefix="deepseek-tests-")
os.environ["HERMES_HOME"] = _session_home.name
os.environ["HERMES_TEST_ISOLATION"] = _session_home.name

# Test-owned source and checked-in speech fixture locations.
REPO = Path(__file__).resolve().parents[1]
AUDIO = REPO / "tests/fixtures/audio"
# Real dispatch configuration; disable Hermes's process-global search memo.
CONFIG = {
    "plugins": {"enabled": ["deepseek-tools"]},
    "web": {"search_backend": "deepseek", "extract_backend": "deepseek", "keyless_rescue": False, "cache_enabled": False},
    "stt": {"enabled": True, "provider": "deepseek"},
    "tts": {"provider": "deepseek"},
}


@pytest.fixture(autouse=True)
def hermes(tmp_path, monkeypatch):
    """Isolate Hermes state and load this repository through real plugin discovery."""
    import hermes_constants as hc
    from agent import secret_scope, transcription_registry, tts_registry, web_search_registry
    from gateway import session_context
    from hermes_cli import env_loader, managed_scope, plugins
    from tools.web_result_cache import search_memo

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(hc, "_get_platform_default_hermes_home", lambda: home)
    monkeypatch.setattr(hc, "_PINNED_PROCESS_HERMES_HOME", None)
    monkeypatch.setattr(managed_scope, "load_managed_env", lambda: {})
    monkeypatch.setattr(env_loader, "get_secret_source_values", lambda *a, **kw: {})
    secret_scope.set_multiplex_active(False)
    token = secret_scope.set_secret_scope(None)
    home_token = hc.set_hermes_home_override(None)
    session_context.reset_session_vars()
    plugins._reset_plugin_managers_for_tests()
    for registry in (web_search_registry, tts_registry, transcription_registry):
        registry._reset_for_tests()
    search_memo.clear()

    def no_network(*args, **kwargs):
        """Reject any test that attempts a real socket connection."""
        raise AssertionError("Tests must not access the network")

    def public_dns(host, port, *args, **kwargs):
        """Supply a public DNS result while retaining Hermes URL-safety decisions."""
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))]

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    requests = []
    responder = [lambda request: httpx.Response(200, json={})]

    def handle(request):
        """Record an HTTP request before invoking the test-owned upstream response."""
        request.read()
        requests.append(request)
        return responder[0](request)

    def configure(target=home, **sections):
        """Write the real profile configuration used by tool dispatch."""
        target.mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(CONFIG))
        for key, value in sections.items():
            config.setdefault(key, {}).update(value)
        (target / "config.yaml").write_text(json.dumps(config))
        return config

    def install(target=home):
        """Symlink the plugin, discover it and inject its HTTP boundary."""
        configure(target)
        directory = target / "plugins"
        directory.mkdir(exist_ok=True)
        (directory / "deepseek-tools").symlink_to(REPO, target_is_directory=True)
        plugins.discover_plugins(force=True)
        manager = plugins.get_plugin_manager()
        info = manager._plugins["deepseek-tools"]
        assert info.enabled, info.error
        web = web_search_registry.get_provider("deepseek")
        assert web is not None
        package = web.__class__.__module__.rsplit(".", 1)[0]
        client = importlib.import_module(package + ".client")
        monkeypatch.setattr(client, "_transport", httpx.MockTransport(handle))
        sleeps = []
        monkeypatch.setattr(client, "_sleep", sleeps.append)
        return SimpleNamespace(web=web, tts=tts_registry.get_provider("deepseek"),
                               stt=transcription_registry.get_provider("deepseek"), client=client,
                               audio=importlib.import_module(package + ".audio"),
                               ocr=importlib.import_module(package + ".ocr"), sleeps=sleeps)

    loaded = install()
    state = SimpleNamespace(home=home, requests=requests, configure=configure, install=install,
                            respond=lambda fn: responder.__setitem__(0, fn), audio_dir=AUDIO, **vars(loaded))
    yield state
    session_context.reset_session_vars()
    secret_scope.set_multiplex_active(False)
    secret_scope.reset_secret_scope(token)
    hc.reset_hermes_home_override(home_token)
    plugins._reset_plugin_managers_for_tests()
    search_memo.clear()


@pytest.fixture
def wav_bytes(hermes):
    """Supply the real PCM WAV fixture for synthesis response tests."""
    return (AUDIO / "pcm16k.wav").read_bytes()
