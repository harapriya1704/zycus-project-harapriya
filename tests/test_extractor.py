"""Tests for the extraction stage (:mod:`src.extractor`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from src.config import Settings
from src.extractor import extract_document_text, transcribe_page
from src.pdf_reader import Document, PdfPage


def _settings(**overrides) -> Settings:
    # Explicit defaults for env-driven knobs so the tests are hermetic.
    kwargs: dict = {
        "extraction_model": "gpt-4o",
        "extraction_temperature": 0.0,
        "text_layer_min_chars": 80,
        "pdf_render_dpi": 72,
        "require_vision": True,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


def test_text_layer_page_returns_verbatim() -> None:
    page = PdfPage(index=0, width_pt=100, height_pt=200, text="INVOICE 123")
    got = transcribe_page(page, chain=None, require_vision=True)
    assert got == "INVOICE 123"


def test_image_page_without_chain_and_require_vision_raises() -> None:
    page = PdfPage(index=0, width_pt=100, height_pt=200, image_path=Path("x.png"))
    with pytest.raises(RuntimeError, match="vision"):
        transcribe_page(page, chain=None, require_vision=True)


def test_image_page_without_chain_is_honest_empty() -> None:
    page = PdfPage(index=0, width_pt=100, height_pt=200, image_path=Path("x.png"))
    got = transcribe_page(page, chain=None, require_vision=False)
    assert got == ""


def test_image_page_uses_vision_chain(tmp_path: Path) -> None:
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    fake = RunnableLambda(lambda page: f"OCR of page {page.index}")
    page = PdfPage(index=0, width_pt=100, height_pt=200, image_path=png)
    assert transcribe_page(page, chain=fake, require_vision=True) == "OCR of page 0"


def test_extract_document_text_mixes_both_paths(tmp_path: Path) -> None:
    png = tmp_path / "p1.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    fake = RunnableLambda(lambda page: "vision text")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[
            PdfPage(index=0, width_pt=100, height_pt=200, text="embedded"),
            PdfPage(index=1, width_pt=100, height_pt=200, image_path=png),
        ],
    )
    extract_document_text(doc, chain=fake, settings=_settings())
    assert doc.pages[0].text == "embedded"
    assert doc.pages[1].text == "vision text"
    assert "embedded\n\nvision text" == doc.text()
