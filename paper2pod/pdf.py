"""PDF paper ingestion: load bytes, build an Anthropic document block, extract metadata.

Unlike markdown/OpenLabs, a PDF is never flattened to a text string. The raw
bytes are handed to Claude natively as a base64 `document` content block, so
the model reads figures, tables, and multi-column layout directly. This is an
Anthropic-only capability, so the PDF path requires transcript.provider ==
"anthropic".
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from paper2pod.logging_setup import ParseError
from paper2pod.parser import (
    PaperMetadata,
    _extract_json_object,
    _normalize_authors,
)

PDF_MAGIC = b"%PDF-"

# Anthropic accepts PDFs up to 32 MB / 100 pages per request. We cap on size
# here so an oversized file fails fast with a clear message instead of a raw
# API error; page count is left to Anthropic to enforce (no PDF library).
DEFAULT_MAX_PDF_MB = 32


def load_pdf(path: Path, max_mb: int = DEFAULT_MAX_PDF_MB) -> bytes:
    """Read a PDF file into bytes, raising ParseError with a clear message otherwise."""
    if not path.exists():
        raise ParseError(f"File not found: {path}", input_file=str(path))
    if path.is_dir():
        raise ParseError(f"Expected a file, got a directory: {path}", input_file=str(path))
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ParseError(f"Could not read file: {e}", input_file=str(path)) from e
    validate_pdf_bytes(raw, max_mb=max_mb, input_file=str(path))
    return raw


def validate_pdf_bytes(
    raw: bytes, max_mb: int = DEFAULT_MAX_PDF_MB, input_file: str | None = None
) -> None:
    """Raise ParseError unless `raw` looks like a PDF within the size cap."""
    if not raw.startswith(PDF_MAGIC):
        raise ParseError(
            "File is not a valid PDF (missing %PDF- header).", input_file=input_file
        )
    if len(raw) > max_mb * 1024 * 1024:
        raise ParseError(
            f"PDF exceeds the {max_mb} MB limit.", input_file=input_file
        )


def build_pdf_document_block(pdf_bytes: bytes) -> dict[str, Any]:
    """Encode raw PDF bytes into an Anthropic base64 document content block."""
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
        },
    }


def extract_pdf_metadata(
    pdf_document: dict[str, Any],
    provider: str,
    model: str,
    secrets: Any,
    fallback_title: str,
) -> PaperMetadata:
    """Ask Claude to read the PDF and return its title/authors.

    Native PDF reading is Anthropic-only. Falls back to `fallback_title`
    (typically the file stem) if the model returns no usable title, so PDF
    ingestion never hard-fails on missing metadata the way the markdown path can.
    """
    if provider != "anthropic":
        raise ParseError(
            "PDF ingestion requires the anthropic transcript provider "
            f"(configured provider is '{provider}')."
        )

    import anthropic

    prompt = (
        "Extract the title and author names from this research paper. "
        'Respond with ONLY compact JSON: {"title": string, "authors": [string, ...]}. '
        "If no clear title exists, use an empty string for title."
    )
    try:
        client = anthropic.Anthropic(api_key=secrets.anthropic_api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [pdf_document, {"type": "text", "text": prompt}],
                }
            ],
        )
        raw = resp.content[0].text
    except Exception as e:
        raise ParseError(f"Metadata extraction failed: {e}") from e

    result = _extract_json_object(raw) or {}
    title = str(result.get("title") or "").strip() or fallback_title
    return PaperMetadata(title=title, authors=_normalize_authors(result.get("authors")))
