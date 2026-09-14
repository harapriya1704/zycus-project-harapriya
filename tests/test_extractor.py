"""Tests for the extraction stage (:mod:`src.extractor`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from src.config import Settings
from src.extractor import (
    _image_fingerprint,
    extract_document_text,
    read_transcription,
    transcribe_page,
    transcription_cache_path,
)
from src.pdf_reader import Document, PdfPage


def _settings(**overrides) -> Settings:
    # Explicit defaults for env-driven knobs so the tests are hermetic.
    kwargs: dict = {
        "vision_model": "gpt-4o",
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
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path)
    assert doc.pages[0].text == "embedded"
    assert doc.pages[1].text == "vision text"
    assert "embedded\n\nvision text" == doc.text()


def _png(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return p


def test_pacing_sleeps_between_vision_calls(tmp_path: Path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: f"OCR of {page.index}")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[
            PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png")),
            PdfPage(index=1, width_pt=100, height_pt=200, image_path=_png(tmp_path, "b.png")),
        ],
    )
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.5), cache_dir=tmp_path)
    assert sleeps == [2.5]  # one wait between the two vision calls, not before the first
    assert [p.text for p in doc.pages] == ["OCR of 0", "OCR of 1"]


def test_no_pacing_on_single_image_page(tmp_path: Path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: "OCR text")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.5), cache_dir=tmp_path)
    assert sleeps == []


def test_no_pacing_when_delay_disabled(tmp_path: Path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: "OCR text")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[
            PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png")),
            PdfPage(index=1, width_pt=100, height_pt=200, image_path=_png(tmp_path, "b.png")),
        ],
    )
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=0.0), cache_dir=tmp_path)
    assert sleeps == []


def test_pacing_ignores_text_layer_pages(tmp_path: Path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: "OCR text")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[
            PdfPage(index=0, width_pt=100, height_pt=200, text="embedded 1"),
            PdfPage(index=1, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png")),
            PdfPage(index=2, width_pt=100, height_pt=200, text="embedded 2"),
            PdfPage(index=3, width_pt=100, height_pt=200, image_path=_png(tmp_path, "b.png")),
        ],
    )
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.0), cache_dir=tmp_path)
    assert sleeps == [2.0]  # only the gap between the two vision calls is paced
    assert [p.text for p in doc.pages] == ["embedded 1", "OCR text", "embedded 2", "OCR text"]


def test_cache_write_then_reuse_skips_api(tmp_path: Path, monkeypatch) -> None:
    png = _png(tmp_path, "page.png")
    calls: list[int] = []

    fake = RunnableLambda(lambda page: calls.append(page.index) or "transcription A")
    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=png)],
    )

    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path / "cache")
    assert doc.pages[0].text == "transcription A"
    assert calls == [0]  # one API call, cached to disk

    cache_dir = tmp_path / "cache"
    assert list(cache_dir.glob("*.txt")), "transcription was not persisted"

    calls.clear()
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=cache_dir)
    assert doc.pages[0].text == "transcription A"
    assert calls == []  # second run served entirely from the cache


def test_cache_key_uses_content_not_byte_length(tmp_path: Path) -> None:
    p1 = _png(tmp_path, "a.png")
    p2 = tmp_path / "b.png"
    # Identical byte length, different content: a length-hashed key would collide.
    p2.write_bytes(p1.read_bytes().replace(b"\x00", b"\x01"))

    assert p1.stat().st_size == p2.stat().st_size  # same length…
    k1 = transcription_cache_path("doc.pdf", 0, _image_fingerprint(p1), tmp_path)
    k2 = transcription_cache_path("doc.pdf", 0, _image_fingerprint(p2), tmp_path)
    assert k1 != k2  # …but content digests keep them apart


def test_read_transcription_treats_empty_or_corrupt_as_miss(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n", encoding="utf-8")
    assert read_transcription(empty) is None

    bogus = tmp_path / "bogus.txt"
    bogus.write_bytes(b"\xff\xfe broken utf-8 \x00\xff")
    assert read_transcription(bogus) is None


def test_empty_cache_file_is_overwritten_by_re_transcription(tmp_path: Path, monkeypatch) -> None:
    png = _png(tmp_path, "page.png")
    calls: list[int] = []
    fake = RunnableLambda(lambda page: calls.append(page.index) or "fresh transcription")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=png)],
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Simulate a hard exit that left a truncated (empty) cache entry behind.
    text_path = transcription_cache_path("doc.pdf", 0, _image_fingerprint(png), cache_dir)
    text_path.write_text("", encoding="utf-8")

    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=cache_dir)
    assert doc.pages[0].text == "fresh transcription"  # re-transcribed, cache ignored
    assert calls == [0]
    assert text_path.read_text(encoding="utf-8") == "fresh transcription"  # healed
    assert not list(cache_dir.glob("*.tmp"))  # atomic write left no temp file behind


def test_cache_hit_skips_pacing_between_vision_calls(tmp_path: Path, monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    p1, p2 = _png(tmp_path, "a.png"), _png(tmp_path, "b.png")
    fake = RunnableLambda(lambda page: "OCR")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[
            PdfPage(index=0, width_pt=100, height_pt=200, image_path=p1),
            PdfPage(index=1, width_pt=100, height_pt=200, image_path=p2),
        ],
    )
    cache_dir = tmp_path / "cache"
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.0), cache_dir=cache_dir)
    assert sleeps == [2.0]

    # Both pages already cached -> no API calls -> no pacing sleeps at all.
    sleeps.clear()
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.0), cache_dir=cache_dir)
    assert sleeps == []
