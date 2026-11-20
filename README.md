# DeepSeek Tools for Hermes Agent

[![CI](https://github.com/wjiec/hermes-deepseek-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/wjiec/hermes-deepseek-tools/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Adds the `deepseek` web search, page extraction, transcription and speech providers,
plus the `deepseek_ocr` attachment tool. Requires Hermes Agent **0.23.3 or later**
and `httpx` (included in Hermes core). Integration tests target v0.23.3.

**Disclosure:** selected tools send search queries, URLs, input audio, speech text
or the local attachments you ask to OCR to `api.deepseek.com`, using the active
profile's DeepSeek account. Successful calls incur DeepSeek charges. OCR uploads
file contents; it does not download URLs. Responses and billing have not been
verified against the live service; tests run offline with simulated HTTP responses.

Install `ffmpeg` (with `ffprobe`) on the gateway host. ASR uploads PCM 16-bit WAV
and Ogg Opus voice notes unchanged when within limits, without needing ffmpeg;
other formats and audio over 120 s / 8 MiB need ffmpeg to convert and split. Hermes uses ffmpeg to convert
synthesized WAV into Opus voice bubbles.

## Install and configure

Plugins are installed per profile: each profile has its own `plugins/` directory,
install record and `plugins.enabled` list. Repeat for every profile the gateway
serves that should use DeepSeek. Pin with `--ref <40-char-sha>` for reproducible
deploys:

```sh
hermes -p default plugins install https://github.com/wjiec/hermes-deepseek-tools --enable
hermes -p <profile> plugins install https://github.com/wjiec/hermes-deepseek-tools --enable
```

Add the key to that profile's `.env`: the default profile uses
`$HERMES_HOME/.env` (normally `~/.hermes/.env`), and a named profile uses
`<Hermes root>/profiles/<profile>/.env`. Keys are never shared between profiles.

```dotenv
DEEPSEEK_API_KEY=your-key
```

Add to each enabled profile's `config.yaml`:

```yaml
web:
  search_backend: deepseek
  extract_backend: deepseek
  keyless_rescue: false
stt:
  enabled: true
  provider: deepseek
tts:
  provider: deepseek
```

For a multiplexed gateway on v0.23.3, also set `web.cache_enabled: false` in each
served profile. Hermes's in-process search memo omits the profile from its cache
key and runs before this provider; disabling it ensures every search checks the
active profile's credentials.

Restart the default gateway after configuring; explicitly select `default` when
another profile is active:

```sh
hermes gateway restart
# Explicit default-profile selection:
hermes -p default gateway restart
```

Upgrade each profile's copy, then restart the default gateway again:

```sh
hermes -p default plugins update deepseek-tools
hermes -p <profile> plugins update deepseek-tools
```

A pinned install (`--ref`) is moved with
`hermes -p <profile> plugins install https://github.com/wjiec/hermes-deepseek-tools --force --ref <new-sha>`.
No runtime dependencies are installed by this plugin.

## Limits and delivery

- Search: non-empty query, at most 128 characters; 1–50 results. Hermes v0.23.3
  buckets tool result counts (for example, 5 becomes an upstream count of 10).
- Fetch: HTTP/HTTPS URLs, no userinfo, at most 8192 characters; four concurrent
  requests, returned in input order. Hermes may further filter or trim results.
- ASR: each upload at most 8 MiB and 120 seconds. Converted audio is 16 kHz mono
  PCM WAV; long input is split at silence midpoints where possible, with segments
  at most 110 seconds. Segment transcripts are joined in order; a failed segment
  fails the whole transcription. Model, language and prompt hints are ignored.
- TTS: `stella` (海星, default, female) or `tide` (暗潮, male). Every request returns
  WAV, regardless of requested output format; speed and model hints are ignored.
  Each request allows at most 10000 billing units (CJK ideographs count as two)
  and a 64 KiB JSON body. Hermes v0.23.3 normally splits plugin speech at 4000
  characters, so a long tool invocation may make multiple billed requests.
  With `voice_compatible=True`, its tool dispatcher converts WAV to Ogg/Opus
  even outside a messaging session. The provider itself returns a `.wav` path.
- OCR: regular local files up to 100 MiB; PDF, DOC/DOCX, PPT/PPTX, PNG, JPEG,
  WebP, GIF, BMP and TIFF. Under multiplexing or non-local terminal backends,
  files must resolve inside the active profile's media cache (including
  `cache/documents`). Hermes secret-file restrictions always apply, including
  through symlinks. Truncated OCR pages are marked in the returned text.

Search, fetch and ASR retry once on transport failures or HTTP
408/429/500/503/504, honoring Retry-After up to five seconds. **TTS and OCR never
retry**, including after a partial response. Long ASR/TTS jobs can have successful
billed segments before a later segment fails.

## Troubleshooting and verification

`DEEPSEEK_API_KEY is not set for profile '<profile>' (add it to <home>/.env)` names
the profile that needs a key. Keys are read at request time through Hermes's
secret scope. A missing named-profile key never falls back to the launch profile
under multiplexing. The v0.23.3 STT tool checks availability first and may show its
generic “STT plugin is not available” message instead of the detailed key error.
An “internal profile-scoping bug” error means Hermes dispatched without a scope;
restart the gateway and investigate the Hermes routing path if it persists.

If ASR requires ffmpeg, install it on the gateway host and check both `ffmpeg`
and `ffprobe` are on PATH. Voice bubbles also require ffmpeg with the Opus encoder.
Invalid or incomplete TTS WAV responses fail and the partial file is removed.

## Live check

`scripts/live_check.py` calls the real DeepSeek API once per capability for each
profile you name, through Hermes' own plugin discovery inside that profile's scope
with multiplexing on. It changes no configuration, prints keys only as 8-character
fingerprints, and costs one search, one fetch, one short TTS, two short ASR and
one one-page OCR per profile. Any `python3` works: the script re-launches itself
inside Hermes' own runtime (found with `hermes --print-runtime-command`):

```sh
python3 scripts/live_check.py --profile default --profile <profile> --json live-check.json
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the test setup. Report security issues privately as described in
[SECURITY.md](SECURITY.md). Changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
