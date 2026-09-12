"""Tests for PDF ingestion (:mod:`src.pdf_reader`)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from src.pdf_reader import PdfError, read_pdf


def _make_pdf(path: Path, *, text_pages: list[str], image_pages: int) -> Path:
    """Create a tiny PDF with embedded text pages and/or blank image pages."""
    doc = pymupdf.open()
    for text in text_pages:
        page = doc.new_page()
        rect = pymupdf.Rect(40, 40, page.rect.width - 40, page.rect.height - 40)
        page.insert_textbox(rect, text, align=pymupdf.TEXT_ALIGN_LEFT)
    for _ in range(image_pages):
        page = doc.new_page()
        page.draw_rect(page.rect, color=(1, 1, 1), fill=(1, 1, 1))
    doc.save(path)
    doc.close()
    return path


def test_read_text_layer(tmp_path: Path) -> None:
    long_text = "Hello invoice. " * 20  # well above the 80-char threshold
    pdf = _make_pdf(tmp_path / "with_text.pdf", text_pages=[long_text, "Line two"], image_pages=0)
    doc = read_pdf(pdf)
    assert doc.filename == "with_text.pdf"
    assert len(doc.pages) == 2
    assert "Hello invoice." in doc.pages[0].text  # wrapped across lines, not exact
    assert doc.pages[0].image_path is None
    assert "Hello invoice." in doc.text()


def test_short_text_layer_is_routed_to_image(tmp_path: Path) -> None:
    # 5 chars < min_text_chars=20 -> treated as image-only page.
    pdf = _make_pdf(tmp_path / "stub.pdf", text_pages=["short"], image_pages=0)
    doc = read_pdf(pdf, min_text_chars=20, dpi=72, render_dir=tmp_path / "pages")
    assert doc.pages[0].text == ""
    assert doc.pages[0].image_path is not None
    assert doc.pages[0].image_path.exists()


def test_image_pages_are_rasterised(tmp_path: Path) -> None:
    pdf = _make_pdf(tmp_path / "scan.pdf", text_pages=[], image_pages=1)
    doc = read_pdf(pdf, dpi=72, render_dir=tmp_path / "pages")
    assert len(doc.pages) == 1
    assert doc.pages[0].text == ""
    assert doc.pages[0].image_path is not None
    assert doc.needs_vision == doc.pages
    assert not doc.has_text_layer


def test_missing_file_raises(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(PdfError):
        read_pdf(tmp_path / "nope.pdf")
