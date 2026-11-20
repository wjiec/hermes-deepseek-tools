# Contributing

Issues and pull requests are welcome.

## Development setup

The tests run against a pinned Hermes Agent checkout, using Hermes' own lockfile:

```sh
git clone --depth 1 --branch v0.23.3 https://github.com/NousResearch/hermes-agent.git ../hermes-agent
(cd ../hermes-agent && UV_PROJECT_ENVIRONMENT="$OLDPWD/.venv" uv sync --frozen --group dev)
.venv/bin/python -m pytest -q
HERMES_HOME="$(mktemp -d)" .venv/bin/hermes plugins validate .
```

`ffmpeg` and `ffprobe` must be on `PATH`. Tests never touch the network: the HTTP boundary is
replaced with `httpx.MockTransport` through `deepseek_tools.client._transport`.

## Guidelines

- Import only Hermes' [public plugin API](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/plugins/public-api.md);
  `hermes plugins validate` must pass without warnings.
- Read `DEEPSEEK_API_KEY` only through `agent.secret_scope.get_secret`, at request time.
- Add a test through the real Hermes entry point for every behaviour change, and a line in
  `CHANGELOG.md` under *Unreleased*.
- Audio fixtures are synthetic; regenerate them with `sh tests/fixtures/audio/generate.sh`.
