"""Unit tests for the PDF ingestion module (paper2pod/pdf.py)."""

import base64

import pytest

from paper2pod.logging_setup import ParseError
from paper2pod.pdf import (
    build_pdf_document_block,
    extract_pdf_metadata,
    load_pdf,
    validate_pdf_bytes,
)

VALID_PDF = b"%PDF-1.7\nfake pdf content"


def test_validate_pdf_bytes_accepts_valid_header():
    validate_pdf_bytes(VALID_PDF)  # no raise


def test_validate_pdf_bytes_rejects_non_pdf():
    with pytest.raises(ParseError, match="not a valid PDF"):
        validate_pdf_bytes(b"hello world")


def test_validate_pdf_bytes_rejects_oversize():
    with pytest.raises(ParseError, match="exceeds"):
        validate_pdf_bytes(VALID_PDF, max_mb=0)


def test_build_pdf_document_block_shape_and_base64():
    block = build_pdf_document_block(VALID_PDF)
    assert block["type"] == "document"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "application/pdf"
    assert base64.standard_b64decode(block["source"]["data"]) == VALID_PDF


def test_load_pdf_missing_file(tmp_path):
    with pytest.raises(ParseError, match="File not found"):
        load_pdf(tmp_path / "nope.pdf")


def test_load_pdf_rejects_non_pdf(tmp_path):
    p = tmp_path / "paper.pdf"
    p.write_bytes(b"not a pdf at all")
    with pytest.raises(ParseError, match="not a valid PDF"):
        load_pdf(p)


def test_load_pdf_reads_valid(tmp_path):
    p = tmp_path / "paper.pdf"
    p.write_bytes(VALID_PDF)
    assert load_pdf(p) == VALID_PDF


def test_extract_pdf_metadata_requires_anthropic():
    block = build_pdf_document_block(VALID_PDF)
    with pytest.raises(ParseError, match="requires the anthropic"):
        extract_pdf_metadata(block, provider="openai", model="gpt", secrets=None, fallback_title="x")
