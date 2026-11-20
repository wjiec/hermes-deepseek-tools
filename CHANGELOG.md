# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-21

### Added
- `deepseek` web search and extract provider (DeepSeek Search and Fetch).
- `deepseek` speech-to-text provider (DeepSeek ASR) with ffmpeg conversion and silence-aware
  splitting for audio over 120 s / 8 MiB.
- `deepseek` text-to-speech provider (DeepSeek TTS, WAV; Hermes converts to Opus voice bubbles).
- `deepseek_ocr` tool for images and documents the user sent.
- Per-profile credentials via Hermes' secret scope, safe on multiplexed gateways.
- `scripts/live_check.py` live acceptance check against the real DeepSeek API.

[Unreleased]: https://github.com/wjiec/hermes-deepseek-tools/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wjiec/hermes-deepseek-tools/releases/tag/v0.1.0
