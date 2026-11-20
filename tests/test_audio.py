import io
import json
import math
import subprocess
import wave
from array import array
from email import policy
from email.parser import BytesParser
from pathlib import Path

import httpx
import pytest


def multipart(request):
    """Parse recorded multipart bytes through the standard MIME parser."""
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + request.headers["content-type"].encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content)
    return {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}


@pytest.mark.parametrize("filename,direct", [("voice.ogg", True), ("pcm16k.wav", True),
                                              ("float_stereo.wav", False), ("clip.m4a", False), ("clip.mp3", False)])
def test_stt_real_tool(hermes, filename, direct):
    """Verify stt real tool."""
    from tools.transcription_tools import transcribe_audio

    hermes.respond(lambda r: httpx.Response(200, json={"text": "  recognized speech  "}))
    source = hermes.audio_dir / filename
    result = transcribe_audio(str(source))
    assert result == {"success": True, "transcript": "recognized speech", "provider": "deepseek"}
    request, = hermes.requests
    assert request.url.path == "/v1/tools/asr" and request.method == "POST"
    parts = multipart(request)
    assert set(parts) == {"file"}
    upload = parts["file"].get_payload(decode=True)
    if direct:
        assert upload == source.read_bytes()
    else:
        assert upload[:4] == b"RIFF" and upload[8:12] == b"WAVE"
        with wave.open(io.BytesIO(upload)) as audio:
            assert (audio.getsampwidth(), audio.getframerate(), audio.getnchannels()) == (2, 16000, 1)
    assert parts["file"].get_content_type() == ("audio/ogg" if filename == "voice.ogg" else "audio/wav")


def test_stt_long_segments_and_cleanup(hermes, monkeypatch):
    """Verify stt long segments and cleanup."""
    from tools.transcription_tools import transcribe_audio

    converted_paths = []
    original_run = hermes.audio._run

    def run(command):
        """Record temporary conversion paths while running the real audio utility."""
        if command[-1].endswith("converted.wav"):
            converted_paths.append(Path(command[-1]))
        return original_run(command)

    monkeypatch.setattr(hermes.audio, "_run", run)
    hermes.respond(lambda r: httpx.Response(200, json={"text": f" segment-{len(hermes.requests)} "}))
    result = transcribe_audio(str(hermes.audio_dir / "long.ogg"))
    assert result["success"]
    assert len(hermes.requests) >= 2
    assert result["transcript"] == " ".join(f"segment-{index}" for index in range(1, len(hermes.requests) + 1))
    duration = 0
    for request in hermes.requests:
        upload = multipart(request)["file"].get_payload(decode=True)
        assert len(upload) <= 8 * 1024 * 1024
        with wave.open(io.BytesIO(upload)) as audio:
            seconds = audio.getnframes() / audio.getframerate()
            assert 0 < seconds <= 110
            duration += seconds
    assert 150 <= duration < 152
    assert converted_paths and all(not path.parent.exists() for path in converted_paths)


def test_stt_upstream_error(hermes):
    """Verify stt upstream error."""
    from tools.transcription_tools import transcribe_audio

    hermes.respond(lambda r: httpx.Response(400, json={"error": {"message": "unrecognized audio"}}))
    result = transcribe_audio(str(hermes.audio_dir / "voice.ogg"))
    assert result == {"success": False, "transcript": "", "provider": "deepseek", "error": "segment 1/1: unrecognized audio"}
    assert len(hermes.requests) == 1


def test_stt_missing_ffmpeg(hermes, monkeypatch):
    """Verify stt missing ffmpeg."""
    from tools.transcription_tools import transcribe_audio

    original = hermes.audio.shutil.which
    monkeypatch.setattr(hermes.audio.shutil, "which", lambda name: None if name == "ffmpeg" else original(name))
    hermes.respond(lambda r: httpx.Response(200, json={"text": "opus works"}))
    assert transcribe_audio(str(hermes.audio_dir / "voice.ogg"))["success"]
    result = transcribe_audio(str(hermes.audio_dir / "clip.mp3"))
    assert not result["success"]
    assert "ffmpeg is required to transcribe mp3 audio with DeepSeek ASR" in result["error"]
    assert len(hermes.requests) == 1


def test_stt_voice_note_without_ffmpeg_or_ffprobe(hermes, monkeypatch):
    """A Telegram-style Ogg Opus voice note needs no ffmpeg/ffprobe; a long one still needs ffmpeg to split."""
    from tools.transcription_tools import transcribe_audio

    monkeypatch.setattr(hermes.audio.shutil, "which", lambda name: None)
    hermes.respond(lambda r: httpx.Response(200, json={"text": "ok"}))
    voice = hermes.audio_dir / "voice.ogg"
    assert transcribe_audio(str(voice))["success"]
    assert len(hermes.requests) == 1
    assert voice.read_bytes() in hermes.requests[0].content
    result = transcribe_audio(str(hermes.audio_dir / "long.ogg"))
    assert not result["success"] and "ffmpeg is required" in result["error"]
    assert len(hermes.requests) == 1


@pytest.mark.parametrize("name", ["voice.ogg", "long.ogg"])
def test_ogg_opus_duration_matches_ffprobe(hermes, name):
    """Granule-position duration agrees with ffprobe within 50 ms."""
    import subprocess
    from pathlib import Path

    path = hermes.audio_dir / name
    probed = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                   "default=nw=1:nk=1", str(path)], capture_output=True, text=True).stdout)
    assert abs(hermes.audio._ogg_opus_seconds(Path(path)) - probed) < 0.05


def test_stt_prefers_last_silence_before_boundary(hermes, tmp_path):
    """Verify stt prefers last silence before boundary."""
    from tools.transcription_tools import transcribe_audio

    source = tmp_path / "silences.wav"
    second = array("h", (int(8000 * math.sin(2 * math.pi * 440 * index / 16000)) for index in range(16000))).tobytes()
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        for index in range(130):
            audio.writeframes(b"\0" * 32000 if 100 <= index < 102 or 107 <= index < 109 else second)
    hermes.respond(lambda r: httpx.Response(200, json={"text": "speech"}))
    assert transcribe_audio(str(source))["success"]
    assert len(hermes.requests) == 2
    first = multipart(hermes.requests[0])["file"].get_payload(decode=True)
    with wave.open(io.BytesIO(first)) as audio:
        assert 107.9 < audio.getnframes() / audio.getframerate() < 108.1


def test_stt_segment_failure_discards_transcript_and_cleans(hermes, monkeypatch):
    """Verify stt segment failure discards transcript and cleans."""
    from tools.transcription_tools import transcribe_audio

    converted = []
    original_run = hermes.audio._run

    def run(command):
        """Record temporary conversion paths while running the real audio utility."""
        if command[-1].endswith("converted.wav"):
            converted.append(Path(command[-1]))
        return original_run(command)

    monkeypatch.setattr(hermes.audio, "_run", run)
    hermes.respond(lambda r: httpx.Response(200, json={"text": "first"}) if len(hermes.requests) == 1 else httpx.Response(400, json={"error": {"message": "failed segment"}}))
    result = transcribe_audio(str(hermes.audio_dir / "long.ogg"))
    assert not result["success"] and result["transcript"] == ""
    assert result["error"].startswith("segment 2/") and "failed segment" in result["error"]
    assert len(hermes.requests) == 2
    assert converted and not converted[0].parent.exists()


@pytest.mark.parametrize("voice,expected", [(None, "stella"), ("tide", "tide"), ("unknown", "stella")])
def test_tts_wav(hermes, wav_bytes, tmp_path, voice, expected):
    """Verify tts wav."""
    hermes.respond(lambda r: httpx.Response(200, content=wav_bytes))
    output = hermes.tts.synthesize("Hello 海星", str(tmp_path / "speech.mp3"), voice=voice, format="opus")
    assert Path(output).suffix == ".wav" and Path(output).read_bytes() == wav_bytes
    request, = hermes.requests
    assert request.method == "POST" and request.url.path == "/v1/tools/tts"
    assert json.loads(request.content) == {"text": "Hello 海星", "voice_id": expected, "format": "wav"}
    assert not hermes.tts.streams_pcm and hermes.tts.voice_compatible


@pytest.mark.parametrize("platform,format", [("telegram", "opus"), ("", "mp3")])
def test_tts_real_delivery_is_opus(hermes, wav_bytes, tmp_path, platform, format):
    """Verify tts real delivery is opus."""
    from gateway.session_context import set_session_vars
    from tools.tts_tool import text_to_speech_tool

    hermes.configure(tts={"voice": "tide", "output_format": format})
    set_session_vars(platform=platform)
    hermes.respond(lambda r: httpx.Response(200, content=wav_bytes))
    result = json.loads(text_to_speech_tool("Hello world", str(tmp_path / "reply.mp3")))
    assert result["success"], result
    output = Path(result["file_path"])
    assert output.suffix == ".ogg" and output.read_bytes().startswith(b"OggS")
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", str(output)], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "opus"
    assert result["voice_compatible"] and "[[audio_as_voice]]" in result["media_tag"]
    assert json.loads(hermes.requests[0].content) == {"text": "Hello world", "voice_id": "tide", "format": "wav"}


@pytest.mark.parametrize("text", ["x" * 10001, "海" * 5001], ids=["ascii", "cjk"])
def test_tts_billing_limit_real_tool(hermes, tmp_path, text):
    """Verify tts billing limit real tool."""
    from tools.tts_tool import text_to_speech_tool

    # Hermes normally splits unknown providers at 4000 chars. Deliver the whole
    # input to the provider to exercise its independent per-request billing cap.
    hermes.configure(tts={"deepseek": {"max_text_length": 20000}})
    result = json.loads(text_to_speech_tool(text, str(tmp_path / "large.mp3")))
    assert not result["success"] and "10000 billing units" in result["error"]
    assert hermes.requests == []


def test_tts_default_tool_chunking(hermes, wav_bytes, tmp_path):
    """Verify tts default tool chunking."""
    from tools.tts_tool import text_to_speech_tool

    hermes.respond(lambda r: httpx.Response(200, content=wav_bytes))
    result = json.loads(text_to_speech_tool("word " * 1000, str(tmp_path / "long.mp3")))
    assert result["success"], result
    assert len(hermes.requests) == result["chunk_count"] == 2
    assert all(len(json.loads(r.content)["text"]) <= 4000 for r in hermes.requests)


@pytest.mark.parametrize("mode", ["riff_size", "data_size", "short_data_size", "premature_close", "invalid_header", "empty_data"])
def test_tts_truncation_cleanup(hermes, wav_bytes, tmp_path, mode):
    """Verify tts truncation cleanup."""
    class BrokenStream(httpx.SyncByteStream):
        """A response body that fails after its first audio bytes."""
        def __iter__(self):
            """Emit partial audio and simulate an upstream protocol failure."""
            yield wav_bytes[:100]
            raise httpx.RemoteProtocolError("peer closed unexpectedly")

    body = bytearray(wav_bytes)
    if mode == "riff_size":
        body = body[:-10]
    elif mode == "data_size":
        body[4:8] = b"\0" * 4
        body = body[:-10]
    elif mode == "short_data_size":
        offset = body.index(b"data")
        size = int.from_bytes(body[offset + 4:offset + 8], "little")
        body[offset + 4:offset + 8] = (size - 100).to_bytes(4, "little")
    elif mode == "invalid_header":
        body[:4] = b"NOPE"
    elif mode == "empty_data":
        offset = body.index(b"data")
        body = body[:offset + 8]
        body[4:8] = b"\0" * 4
        body[offset + 4:offset + 8] = b"\xff" * 4
    hermes.respond(lambda r: httpx.Response(200, stream=BrokenStream()) if mode == "premature_close" else httpx.Response(200, content=bytes(body)))
    output = tmp_path / "partial.mp3"
    with pytest.raises(hermes.client.DeepSeekError):
        hermes.tts.synthesize("hello", str(output))
    assert not output.with_suffix(".wav").exists()
    assert len(hermes.requests) == 1


def test_tts_valid_trailing_chunk(hermes, wav_bytes, tmp_path):
    """Verify tts valid trailing chunk."""
    body = bytearray(wav_bytes)
    body.extend(b"JUNK\x04\x00\x00\x00test")
    body[4:8] = (len(body) - 8).to_bytes(4, "little")
    hermes.respond(lambda r: httpx.Response(200, content=bytes(body)))
    output = hermes.tts.synthesize("hello", str(tmp_path / "trailing.mp3"))
    assert Path(output).read_bytes() == body


@pytest.mark.parametrize("sentinel", [0, 0xFFFFFFFF])
def test_tts_streaming_size_sentinel(hermes, wav_bytes, tmp_path, sentinel):
    """Verify tts streaming size sentinel."""
    body = bytearray(wav_bytes)
    body[4:8] = sentinel.to_bytes(4, "little")
    offset = body.index(b"data")
    body[offset + 4:offset + 8] = sentinel.to_bytes(4, "little")
    hermes.respond(lambda r: httpx.Response(200, content=bytes(body)))
    output = hermes.tts.synthesize("hello", str(tmp_path / "sentinel.mp3"))
    assert Path(output).read_bytes() == body


def test_tts_truncated_real_tool(hermes, wav_bytes, tmp_path):
    """Verify tts truncated real tool."""
    from tools.tts_tool import text_to_speech_tool

    hermes.respond(lambda r: httpx.Response(200, content=wav_bytes[:-10]))
    output = tmp_path / "broken.mp3"
    result = json.loads(text_to_speech_tool("hello", str(output)))
    assert not result["success"] and "truncated" in result["error"]
    assert not output.with_suffix(".wav").exists()
    assert len(hermes.requests) == 1
