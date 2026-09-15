"""Tests for the extraction stage (:mod:`src.extractor`)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda
from src.config import Settings
from src.extractor import (
    _data_uri,
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
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.5, vision_batch_pause=0.0), cache_dir=tmp_path)
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
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=0.0, vision_batch_pause=0.0), cache_dir=tmp_path)
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
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.0, vision_batch_pause=0.0), cache_dir=tmp_path)
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
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.0, vision_batch_pause=0.0), cache_dir=cache_dir)
    assert sleeps == [2.0]

    # Both pages already cached -> no API calls -> no pacing sleeps at all.
    sleeps.clear()
    extract_document_text(doc, chain=fake, settings=_settings(vision_pacing_delay=2.0, vision_batch_pause=0.0), cache_dir=cache_dir)
    assert sleeps == []


# ---------------------------------------------------------------------------
# Batched vision processing + blank-page skipping (precision mode)
# ---------------------------------------------------------------------------

def test_batch_pause_applied_between_batches(tmp_path: Path, monkeypatch) -> None:
    """Vision calls are grouped in batches; a batch pause is inserted at
    batch boundaries on top of the per-page pacing."""
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: f"OCR {page.index}")

    pages = [
        PdfPage(index=i, width_pt=100, height_pt=200, image_path=_png(tmp_path, f"p{i}.png"))
        for i in range(4)
    ]
    doc = Document(path=tmp_path / "doc.pdf", pages=pages)
    extract_document_text(
        doc,
        chain=fake,
        settings=_settings(vision_pacing_delay=2.0, vision_batch_size=2, vision_batch_pause=1.0),
        cache_dir=tmp_path / "cache",
    )
    assert [p.text for p in doc.pages] == [f"OCR {i}" for i in range(4)]
    # 4 calls: per-page pacing before calls 2..4 (3 x 2.0) plus a 1.0 batch
    # pause before call 3 (the start of batch 2).
    assert sleeps == [2.0, 1.0, 2.0, 2.0]


def test_pacing_unchanged_when_batch_pause_disabled(tmp_path: Path, monkeypatch) -> None:
    """Disabling the batch pause preserves plain per-page pacing."""
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: "OCR")

    pages = [
        PdfPage(index=i, width_pt=100, height_pt=200, image_path=_png(tmp_path, f"p{i}.png"))
        for i in range(3)
    ]
    doc = Document(path=tmp_path / "doc.pdf", pages=pages)
    extract_document_text(
        doc,
        chain=fake,
        settings=_settings(vision_pacing_delay=2.0, vision_batch_size=2, vision_batch_pause=0.0),
        cache_dir=tmp_path / "cache",
    )
    assert sleeps == [2.0, 2.0]


def test_default_pacing_applies_batch_pause_per_page(tmp_path: Path, monkeypatch) -> None:
    """With the default single-page batch, the 2s batch pause separates every
    page request (pacing applies between consecutive pages too)."""
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: "OCR")

    pages = [
        PdfPage(index=i, width_pt=100, height_pt=200, image_path=_png(tmp_path, f"p{i}.png"))
        for i in range(3)
    ]
    doc = Document(path=tmp_path / "doc.pdf", pages=pages)
    extract_document_text(
        doc,
        chain=fake,
        settings=_settings(vision_pacing_delay=2.0),
        cache_dir=tmp_path / "cache",
    )
    # Batch pause (2.0, default) fires at every batch boundary, i.e. before
    # every call after the first, then per-page pacing (2.0) adds on top.
    assert sleeps == [2.0, 2.0, 2.0, 2.0]


def test_blank_page_skips_vision_call(tmp_path: Path) -> None:
    """A near-blank scanned page must not spend vision tokens."""
    pytest.importorskip("PIL")
    import PIL.Image

    blank = tmp_path / "blank.png"
    PIL.Image.new("L", (200, 300), 255).save(blank)  # all white

    calls: list[int] = []
    fake = RunnableLambda(lambda page: calls.append(page.index) or "CONTENT")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=blank)],
    )
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path / "cache")
    assert calls == []  # vision chain never invoked for a blank page
    assert doc.pages[0].text == ""


def test_blank_skip_disable_forces_vision_on_blank_page(tmp_path: Path) -> None:
    """When blank-page skipping is disabled, even a blank page goes to vision."""
    pytest.importorskip("PIL")
    import PIL.Image

    blank = tmp_path / "blank.png"
    PIL.Image.new("L", (200, 300), 255).save(blank)

    kinds: list[str] = []
    fake = RunnableLambda(lambda page: kinds.append("vision") or "CONTENT")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=blank)],
    )
    extract_document_text(
        doc,
        chain=fake,
        settings=_settings(vision_skip_blank_pages=False),
        cache_dir=tmp_path / "cache",
    )
    assert kinds == ["vision"]
    assert doc.pages[0].text == "CONTENT"


def test_non_blank_page_still_transcribed(tmp_path: Path) -> None:
    """A page with visible ink is still sent through the vision chain."""
    pytest.importorskip("PIL")
    import PIL.Image
    import PIL.ImageDraw

    ink = tmp_path / "ink.png"
    img = PIL.Image.new("L", (200, 300), 255)
    PIL.ImageDraw.Draw(img).rectangle([10, 10, 190, 100], fill=0)
    img.save(ink)

    fake = RunnableLambda(lambda page: f"OCR {page.index}")
    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=ink)],
    )
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path / "cache")
    assert doc.pages[0].text == "OCR 0"


# ---------------------------------------------------------------------------
# Local OCR fallback (precision mode) + JPEG payload compression
# ---------------------------------------------------------------------------

def test_precision_ocr_fallback_skips_vision(tmp_path: Path, monkeypatch) -> None:
    """Readable scanned pages are transcribed by local OCR, never sent to the
    Vision API (zero TPM spend)."""
    vision_calls: list[int] = []
    ocr_text = "INVOICE NR 265345 DATED 2026-02-02 supplier Phocus TOTAL 438.00 EUR"  # >=50 chars
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: ocr_text)

    fake = RunnableLambda(lambda page: vision_calls.append(page.index) or "VISION")
    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path / "cache")
    assert vision_calls == []  # OCR provided text -> vision never invoked
    assert doc.pages[0].text == ocr_text


def test_precision_ocr_fallback_is_cached(tmp_path: Path, monkeypatch) -> None:
    """OCR output is persisted, so a re-run serves from cache (no re-OCR)."""
    ocr_paths: list[Path] = []
    ocr_text = "INVOICE NR 265345 DATED 2026-02-02 supplier Phocus TOTAL 438.00 EUR"
    monkeypatch.setattr("src.extractor._local_ocr", lambda p: ocr_paths.append(p) or ocr_text)
    fake = RunnableLambda(lambda page: "VISION")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    cache_dir = tmp_path / "cache"
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=cache_dir)
    assert doc.pages[0].text == ocr_text
    assert len(ocr_paths) == 1  # OCR ran once, result cached to disk

    ocr_paths.clear()
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=cache_dir)
    assert doc.pages[0].text == ocr_text
    assert ocr_paths == []  # served from the transcription cache, no re-OCR


def test_precision_short_ocr_is_rejected_and_vision_used(tmp_path: Path, monkeypatch) -> None:
    """Short OCR (<50 chars) counts as "no readable text": the page is sent to
    the Vision API, not trusted for structuring."""
    vision_calls: list[int] = []
    monkeypatch.setattr(
        "src.extractor._local_ocr", lambda _image_path: "stamp only"  # 10 chars
    )
    fake = RunnableLambda(lambda page: vision_calls.append(page.index) or "VISION-TEXT")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path / "cache")
    assert vision_calls == [0]  # short OCR ignored -> vision invoked
    assert doc.pages[0].text == "VISION-TEXT"


def test_precision_ocr_failure_falls_through_to_vision(tmp_path: Path, monkeypatch) -> None:
    """When local OCR yields nothing the page still goes to the vision chain."""
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: None)
    fake = RunnableLambda(lambda page: "VISION-TEXT")

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(doc, chain=fake, settings=_settings(), cache_dir=tmp_path / "cache")
    assert doc.pages[0].text == "VISION-TEXT"


# ---------------------------------------------------------------------------
# Manual Vision-LLM trigger (use_llm / --use-llm): default run is OCR-only
# ---------------------------------------------------------------------------

def test_default_run_leaves_unreadable_page_empty(tmp_path: Path, monkeypatch) -> None:
    """Without the manual LLM trigger the Vision LLM never runs: an image-only
    page that easyocr cannot read stays honest-empty (no raise, no tokens)."""
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: None)
    built: list[object] = []
    monkeypatch.setattr("src.extractor._chain_for", lambda *a, **k: built.append(a) or None)

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(doc, chain=None, settings=_settings(), cache_dir=tmp_path / "cache")
    assert built == []  # no vision chain built, even though require_vision=True
    assert doc.pages[0].text == ""


def test_credentials_alone_do_not_trigger_vision(tmp_path: Path, monkeypatch) -> None:
    """A configured key never activates the Vision LLM by itself; only the
    manual ``use_llm`` trigger does."""
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: None)
    built: list[object] = []
    monkeypatch.setattr("src.extractor._chain_for", lambda *a, **k: built.append(a) or None)

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(
        doc,
        chain=None,
        settings=_settings(hf_token="hf-test"),
        cache_dir=tmp_path / "cache",
    )
    assert built == []  # key present but no --use-llm -> still fully local
    assert doc.pages[0].text == ""


def test_use_llm_trigger_routes_unreadable_pages_to_vision(tmp_path: Path, monkeypatch) -> None:
    """With use_llm=True, pages easyocr cannot read are sent through the
    Vision LLM (chain auto-built from the configured key)."""
    vision_calls: list[int] = []
    fake = RunnableLambda(lambda page: vision_calls.append(page.index) or "VISION-TEXT")
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: None)
    monkeypatch.setattr("src.extractor._chain_for", lambda *a, **k: fake)

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(
        doc,
        chain=None,
        settings=_settings(use_llm=True, hf_token="hf-test"),
        cache_dir=tmp_path / "cache",
    )
    assert vision_calls == [0]
    assert doc.pages[0].text == "VISION-TEXT"


def test_use_llm_without_credentials_raises_when_vision_required(tmp_path: Path, monkeypatch) -> None:
    """Triggering the Vision LLM without a configured key surfaces honestly
    (never silently returns empty for a page that needs vision)."""
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    with pytest.raises(RuntimeError, match="no vision chain/API key"):
        extract_document_text(
            doc,
            chain=None,
            settings=_settings(use_llm=True, hf_token="", llm_api_key=""),
            cache_dir=tmp_path / "cache",
        )


def test_data_uri_compresses_page_to_downscaled_jpeg(tmp_path: Path) -> None:
    """Vision payloads are downscaled to <=1000px and JPEG-compressed instead
    of shipping full-resolution PNGs (4-5x smaller payloads)."""
    pytest.importorskip("PIL")
    import io

    import PIL.Image

    # Noise is the hardest case for PNG (it barely compresses), so any size
    # drop proves the downscaling + JPEG path actually shrinks the payload.
    src_png = tmp_path / "src.png"
    PIL.Image.effect_noise((1600, 2100), 40).convert("RGB").save(src_png)

    uri = _data_uri(src_png)
    assert uri.startswith("data:image/jpeg;base64,")
    payload = base64.b64decode(uri.split(",", 1)[1])
    assert payload[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert len(payload) < src_png.stat().st_size  # smaller than the source PNG

    img = PIL.Image.open(io.BytesIO(payload))
    assert max(img.size) <= 1000  # longest edge capped


def test_data_uri_falls_back_to_png_when_pillow_missing(tmp_path: Path, monkeypatch) -> None:
    """Without Pillow, the raw PNG data-URI is produced (ever-broken pipeline)."""
    import builtins

    real_import = builtins.__import__

    def _block_pil(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pil)
    png = _png(tmp_path, "raw.png")
    uri = _data_uri(png)
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == png.read_bytes()
