"""DeepSeek Tools — Hermes plugin entry point.

Registers one provider id, ``deepseek``, for web search/extract, speech-to-text and
text-to-speech, plus the ``deepseek_ocr`` tool. Every request reads the active
profile's ``DEEPSEEK_API_KEY`` at call time (``agent.secret_scope.get_secret``).
"""
from __future__ import annotations


def register(ctx) -> None:
    """Register the DeepSeek providers and attachment tool with Hermes."""
    from .deepseek_tools.ocr import OCR_SCHEMA, OCR_TOOL_NAME, OCR_TOOLSET, check_ocr_available, handle_ocr
    from .deepseek_tools.stt import DeepSeekTranscriptionProvider
    from .deepseek_tools.tts import DeepSeekTTSProvider
    from .deepseek_tools.web import DeepSeekWebSearchProvider

    ctx.register_web_search_provider(DeepSeekWebSearchProvider())
    ctx.register_transcription_provider(DeepSeekTranscriptionProvider())
    ctx.register_tts_provider(DeepSeekTTSProvider())
    ctx.register_tool(
        name=OCR_TOOL_NAME,
        toolset=OCR_TOOLSET,
        schema=OCR_SCHEMA,
        handler=handle_ocr,
        check_fn=check_ocr_available,
        description=OCR_SCHEMA["description"],
        emoji="📄",
    )
