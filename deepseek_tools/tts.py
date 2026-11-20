"""Billed WAV synthesis; Hermes owns voice-bubble conversion to Opus."""
from __future__ import annotations

import json
import struct
from pathlib import Path

from agent.tts_provider import TTSProvider

from .client import DeepSeekError, is_available, stream_tts


def _billing_length(text: str) -> int:
    """Count CJK ideographs as two billing units and other characters as one."""
    return sum(2 if (0x3400 <= ord(char) <= 0x4DBF or 0x4E00 <= ord(char) <= 0x9FFF
                     or 0xF900 <= ord(char) <= 0xFAFF or 0x20000 <= ord(char) <= 0x323AF) else 1 for char in text)


def _validate_wav(path: Path):
    """Check advertised RIFF/data lengths, accepting streaming size sentinels."""
    actual = path.stat().st_size
    with path.open("rb") as source:
        header = source.read(12)
        if len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise DeepSeekError("DeepSeek TTS did not return RIFF/WAVE audio")
        riff_size = struct.unpack_from("<I", header, 4)[0]
        if riff_size not in {0, 0xFFFFFFFF} and riff_size + 8 != actual:
            raise DeepSeekError("DeepSeek TTS WAV is truncated (RIFF size mismatch)")

        found_data = False
        while chunk := source.read(8):
            if len(chunk) != 8:
                raise DeepSeekError("DeepSeek TTS WAV has an incomplete chunk header")
            kind, size = struct.unpack("<4sI", chunk)
            available = actual - source.tell()
            if kind == b"data":
                if available < 1:
                    raise DeepSeekError("DeepSeek TTS WAV is truncated (data size mismatch)")
                if size in {0, 0xFFFFFFFF}:
                    return
                found_data = True
            if size + size % 2 > available:
                raise DeepSeekError("DeepSeek TTS WAV is truncated (chunk size mismatch)")
            source.seek(size + size % 2, 1)
        if found_data:
            return
    raise DeepSeekError("DeepSeek TTS WAV contains no audio data")


class DeepSeekTTSProvider(TTSProvider):
    """Hermes WAV synthesis with voice-bubble conversion enabled."""
    # Stable registry identity and picker label.
    name = "deepseek"
    display_name = "DeepSeek"

    @property
    def voice_compatible(self):
        """Opt into Hermes conversion of WAV into Opus voice bubbles."""
        return True

    def is_available(self):
        """Probe profile credentials without network access or exceptions."""
        return is_available()

    def list_voices(self):
        """List Stella first as the default voice, followed by Tide."""
        return [{"id": "stella", "display": "海星", "gender": "female"},
                {"id": "tide", "display": "暗潮", "gender": "male"}]

    def list_models(self):
        """Expose the single DeepSeek model supported by this provider."""
        return [{"id": "deepseek-tts", "display": "DeepSeek TTS"}]

    def synthesize(self, text, output_path, *, voice=None, model=None, speed=None, format="mp3", **extra):
        """Stream one billed WAV response, validate it and remove incomplete output."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("DeepSeek TTS text must be non-empty")
        if _billing_length(text) > 10000:
            raise ValueError("DeepSeek TTS text exceeds 10000 billing units (CJK ideographs count as 2)")

        body = {"text": text, "voice_id": voice if voice in {"stella", "tide"} else "stella", "format": "wav"}
        if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) > 64 * 1024:
            raise ValueError("DeepSeek TTS request body exceeds 64 KiB")

        output = Path(output_path).with_suffix(".wav")
        opened = False
        try:
            with stream_tts(body) as response:
                with output.open("wb") as target:
                    opened = True
                    for chunk in response.iter_bytes():
                        target.write(chunk)
            _validate_wav(output)
            return str(output)
        except Exception as exc:
            if opened:
                output.unlink(missing_ok=True)
            if isinstance(exc, OSError):
                raise DeepSeekError("Could not write DeepSeek TTS audio") from None
            raise
