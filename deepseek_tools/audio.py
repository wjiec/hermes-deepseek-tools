"""ASR container inspection and temporary PCM conversion/segmentation."""
from __future__ import annotations

import re
import shutil
import struct
import subprocess
import wave
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .client import DeepSeekError

# DeepSeek upload limits, in bytes and seconds respectively.
MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 120
# Converted segments leave headroom beneath the API duration limit.
SEGMENT_SECONDS = 110


def sniff(path: Path) -> str:
    """Identify PCM16 WAV or first-packet Ogg Opus from container bytes."""
    with path.open("rb") as source:
        header = source.read(12)
        if header[:4] == b"OggS":
            source.seek(0)
            page = source.read(27)
            if len(page) == 27:
                lacing = source.read(page[26])
                if lacing and source.read(8) == b"OpusHead":
                    return "opus"
        if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
            while chunk := source.read(8):
                if len(chunk) != 8:
                    break
                kind, size = struct.unpack("<4sI", chunk)
                if kind == b"fmt ":
                    fmt = source.read(min(size, 16))
                    if len(fmt) == 16 and struct.unpack_from("<H", fmt)[0] == 1 and struct.unpack_from("<H", fmt, 14)[0] == 16:
                        return "wav16"
                    return "other"
                source.seek(size + size % 2, 1)
    return "other"


def _run(command):
    """Run one audio utility and translate a nonzero exit without exposing its output."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise DeepSeekError(f"{Path(command[0]).name} could not process the audio")
    return result


def _ogg_opus_seconds(path: Path) -> float | None:
    """Read Opus duration from the last Ogg page's granule position (48 kHz clock minus pre-skip)."""
    with path.open("rb") as source:
        head = source.read(64 * 1024)
        start = head.find(b"OpusHead")
        if start < 0 or len(head) < start + 12:
            return None
        pre_skip = struct.unpack_from("<H", head, start + 10)[0]
        size = path.stat().st_size
        source.seek(max(0, size - 65536))
        tail = source.read()
    page = tail.rfind(b"OggS")
    if page < 0 or len(tail) < page + 14:
        return None
    granule = struct.unpack_from("<q", tail, page + 6)[0]
    seconds = (granule - pre_skip) / 48000
    return seconds if seconds > 0 else None


def duration(path: Path, kind: str) -> float:
    """Measure seconds from the WAV header, the Ogg Opus granule position, or ffprobe."""
    if kind == "wav16":
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    # Telegram/WhatsApp voice notes are Ogg Opus; they must not need ffprobe.
    if kind == "opus" and (seconds := _ogg_opus_seconds(path)) is not None:
        return seconds
    probe = shutil.which("ffprobe")
    if not probe:
        raise DeepSeekError("ffprobe is required to measure audio duration for DeepSeek ASR")
    result = _run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    seconds = float(result.stdout.strip())
    if not 0 < seconds < float("inf"):
        raise DeepSeekError("Audio duration must be finite and positive")
    return seconds


def _silence_midpoints(ffmpeg: str, path: Path) -> list[float]:
    """Read ffmpeg silence intervals as candidate split times in seconds."""
    result = _run([ffmpeg, "-hide_banner", "-i", str(path), "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"])
    start = None
    midpoints = []
    for kind, value in re.findall(r"silence_(start|end):\s*([\d.]+)", result.stderr):
        if kind == "start":
            start = float(value)
        elif start is not None:
            midpoints.append((start + float(value)) / 2)
            start = None
    return midpoints


@contextmanager
def prepare_audio(file_path):
    """Yield acceptable upload paths and clean all temporary conversion files."""
    path = Path(file_path)
    kind = sniff(path)
    seconds = duration(path, kind)
    if kind in {"wav16", "opus"} and path.stat().st_size <= MAX_BYTES and seconds <= MAX_SECONDS:
        yield [(path, "audio/wav" if kind == "wav16" else "audio/ogg")]
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DeepSeekError(f"ffmpeg is required to transcribe {path.suffix.lstrip('.') or 'unknown'} audio with DeepSeek ASR")
    with TemporaryDirectory(prefix="deepseek-asr-") as directory:
        converted = Path(directory) / "converted.wav"
        _run([ffmpeg, "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(converted)])

        cuts = _silence_midpoints(ffmpeg, converted) if seconds > SEGMENT_SECONDS else []
        segments = []
        with wave.open(str(converted), "rb") as source:
            rate, total = source.getframerate(), source.getnframes()
            start = 0
            while start < total:
                boundary = min(start + SEGMENT_SECONDS * rate, total)
                candidates = [int(point * rate) for point in cuts if start < int(point * rate) <= boundary]
                end = max(candidates) if boundary < total and candidates else boundary
                segment = Path(directory) / f"segment-{len(segments) + 1}.wav"
                with wave.open(str(segment), "wb") as target:
                    target.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                    source.setpos(start)
                    target.writeframes(source.readframes(end - start))
                segments.append((segment, "audio/wav"))
                start = end

        yield segments
