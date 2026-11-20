"""File transcription with sequential, bounded DeepSeek ASR uploads."""
from __future__ import annotations

from agent.transcription_provider import TranscriptionProvider

from .audio import prepare_audio
from .client import is_available, request_json


class DeepSeekTranscriptionProvider(TranscriptionProvider):
    """Hermes file transcription backed by DeepSeek ASR."""
    # Stable registry identity and picker label.
    name = "deepseek"
    display_name = "DeepSeek"

    def is_available(self):
        """Probe profile credentials without network access or exceptions."""
        return is_available()

    def list_models(self):
        """Expose the single DeepSeek model supported by this provider."""
        return [{"id": "deepseek-asr", "display": "DeepSeek ASR"}]

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        """Transcribe sequential uploads and convert every failure into the STT envelope."""
        try:
            with prepare_audio(file_path) as segments:
                transcripts = []
                for index, (path, mime) in enumerate(segments, 1):
                    try:
                        extension = ".wav" if mime == "audio/wav" else ".ogg"
                        result = request_json("asr", files={"file": (path.stem + extension, path.read_bytes(), mime)})
                        text = result["text"]
                        if not isinstance(text, str):
                            raise ValueError("DeepSeek ASR returned invalid text")
                        transcripts.append(text.strip())
                    except Exception as exc:
                        return {"success": False, "transcript": "", "provider": self.name,
                                "error": f"segment {index}/{len(segments)}: {exc}"}
            return {"success": True, "transcript": " ".join(transcripts), "provider": self.name}
        except Exception as exc:
            return {"success": False, "transcript": "", "provider": self.name, "error": str(exc)}
