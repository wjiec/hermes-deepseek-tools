"""Attachment-only OCR with Hermes secret-file and profile media boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
from agent.file_safety import get_read_block_error
from agent.secret_scope import UnscopedSecretError, is_multiplex_active
from hermes_constants import get_hermes_home
from tools.registry import tool_error

from .client import DeepSeekError, is_available, request_json

# Stable tool and toolset identities used by Hermes registration.
OCR_TOOL_NAME = "deepseek_ocr"
OCR_TOOLSET = "deepseek"
# OCR modes accepted by the upstream API.
_PURPOSES = {"text_recognition", "rich_layout_recognition"}
# The public Hermes function schema points agents at user-supplied attachments.
OCR_SCHEMA = {
    "name": OCR_TOOL_NAME,
    "description": "Extract text from an image or document the user sent (PDF, DOC/DOCX, PPT/PPTX, PNG, JPEG, WebP, GIF, BMP, TIFF) using DeepSeek OCR. Pass the local path shown in the attachment note.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Local attachment path or file:// URI"},
        "purpose": {"type": "string", "enum": sorted(_PURPOSES), "default": "text_recognition"},
    }, "required": ["path"]},
}


def check_ocr_available():
    """Expose the non-raising credential probe to tool discovery."""
    return is_available()


def allowed_location(resolved: Path) -> bool:
    """Limit multiplexed or remote-terminal reads to the profile media roots."""
    home = get_hermes_home()
    roots = [home / directory for directory in (
        "cache", "images", "image_cache", "audio_cache", "video_cache", "document_cache",
        "temp_vision_images", "temp_video_files", "cache/documents",
    )]
    if any(resolved.is_relative_to(root.resolve()) for root in roots):
        return True
    return not is_multiplex_active() and os.environ.get("TERMINAL_ENV", "local").strip().lower() in {"", "local"}


def _attachment(path) -> Path:
    """Resolve and validate a local document before any upload can begin."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty local attachment path")
    parsed = urlsplit(path)
    if parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            raise ValueError("Only local paths and file:// URIs are accepted")
        path = unquote(parsed.path)

    resolved = Path(path).expanduser().absolute().resolve(strict=True)
    blocked = get_read_block_error(str(resolved))
    if blocked:
        raise ValueError(blocked)
    if not allowed_location(resolved):
        raise ValueError("OCR path must be inside this profile's media cache")

    if not resolved.is_file():
        raise ValueError("OCR path must be a regular file")
    if resolved.stat().st_size > 100 * 1024 * 1024:
        raise ValueError("OCR file exceeds 100 MiB")

    with resolved.open("rb") as source:
        magic = source.read(12)
    supported = (magic.startswith((b"%PDF", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"BM", b"II*\0", b"MM\0*", b"\xd0\xcf\x11\xe0"))
                 or (magic.startswith(b"RIFF") and magic[8:12] == b"WEBP")
                 or (magic.startswith(b"PK\x03\x04") and resolved.suffix.lower() in {".docx", ".pptx"}))
    if not supported:
        raise ValueError("Unsupported OCR file format")
    return resolved


def handle_ocr(args, **kw):
    """Upload a validated attachment and serialize page text or a tool failure."""
    try:
        path = _attachment(args.get("path"))
        purpose = args.get("purpose", "text_recognition")
        if purpose not in _PURPOSES:
            raise ValueError("Invalid OCR purpose")

        with path.open("rb") as source:
            result = request_json("ocr", files={"file": (path.name, source, "application/octet-stream")},
                                  data={"purpose": purpose}, timeout=httpx.Timeout(connect=10, read=190, write=130, pool=10))

        pages = []
        for index, page in enumerate(result["pages"], 1):
            text = page["text"]
            if not isinstance(text, str):
                raise DeepSeekError("DeepSeek OCR returned invalid page text")
            pages.append({"index": page.get("index", index), "text": text, "truncated": page.get("finish_reason") == "length"})

        parts = []
        for index, page in enumerate(pages, 1):
            text = (f"[page {index}]\n" if len(pages) > 1 else "") + page["text"]
            if page["truncated"]:
                text += "\n[truncated]"
            parts.append(text)
        return json.dumps({"success": True, "page_count": len(pages), "purpose": purpose,
                           "text": "\n\n---\n\n".join(parts), "pages": pages}, ensure_ascii=False)
    except UnscopedSecretError:
        raise
    except Exception as exc:
        return tool_error(str(exc), success=False)
