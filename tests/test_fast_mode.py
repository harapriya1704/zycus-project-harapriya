"""Fast Testing Mode: vision bypass, truncation, compact prompt, zero retries."""

from __future__ import annotations

from pathlib import Path

import pymupdf
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from src.config import Settings
from src.extractor import extract_document_text
from src.formatter import (
    _STRUCTURING_SYSTEM_PROMPT,
    _STRUCTURING_SYSTEM_PROMPT_FAST,
    MAX_STRUCTURING_CHARS,
    build_autodraft_chain,
    format_autodraft,
)
from src.master_data import MasterData
from src.pdf_reader import Document, PdfPage, read_pdf
from src.schemas import FileOutput


def _text_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, page.rect.width - 40, page.rect.height - 40), text)
    doc.save(path)
    doc.close()
    return path


def _png(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return p


# ---------------------------------------------------------------------------
# 1. Text-first extraction (bypass vision)
# ---------------------------------------------------------------------------


def test_read_pdf_fast_threshold_keeps_text_pages_text(tmp_path: Path) -> None:
    """Pages with >50 printable chars are NOT rasterised under the fast rule."""
    pdf = _text_pdf(tmp_path / "fast.pdf", "Q" * 60)
    fast_doc = read_pdf(pdf, min_text_chars=50, dpi=72, render_dir=tmp_path / "fast_render")
    assert fast_doc.pages[0].image_path is None  # text layer reused, no PNG
    assert fast_doc.pages[0].text.strip()

    precise_doc = read_pdf(pdf, min_text_chars=80, dpi=72, render_dir=tmp_path / "precise_render")
    assert precise_doc.pages[0].image_path is not None  # same page rasterised


def test_extract_fast_mode_hard_bypasses_vision_chain(tmp_path: Path, monkeypatch) -> None:
    """Fast mode NEVER invokes the vision chain; image pages use local OCR."""
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    vision_calls: list[int] = []
    fake = RunnableLambda(lambda page: vision_calls.append(page.index) or "VISION-OUTPUT")

    def _fake_ocr(image_path: Path) -> str:
        return f"OCR of {image_path.name}"

    monkeypatch.setattr("src.extractor._local_ocr", _fake_ocr)

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[
            PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png")),
            PdfPage(index=1, width_pt=100, height_pt=200, image_path=_png(tmp_path, "b.png")),
        ],
    )
    extract_document_text(
        doc,
        chain=fake,
        settings=Settings(fast_mode=True, vision_pacing_delay=9.0),
        cache_dir=tmp_path / "cache",
    )
    assert sleeps == []  # pacing disabled in fast mode
    assert vision_calls == []  # hard bypass: vision chain never invoked
    assert [p.text for p in doc.pages] == ["OCR of a.png", "OCR of b.png"]


def test_fast_mode_ocr_empty_still_skips_vision_chain(tmp_path: Path, monkeypatch) -> None:
    """Even when local OCR yields nothing, fast mode must not fall to vision."""
    vision_calls: list[int] = []
    fake = RunnableLambda(lambda page: vision_calls.append(page.index) or "VISION-OUTPUT")
    # OCR produces no text: the old bug fell through to the vision chain here.
    monkeypatch.setattr("src.extractor._local_ocr", lambda _image_path: None)

    doc = Document(
        path=tmp_path / "doc.pdf",
        pages=[PdfPage(index=0, width_pt=100, height_pt=200, image_path=_png(tmp_path, "a.png"))],
    )
    extract_document_text(
        doc,
        chain=fake,
        settings=Settings(fast_mode=True),
        cache_dir=tmp_path / "cache",
    )
    assert vision_calls == []  # vision chain never invoked, even on empty OCR
    assert doc.pages[0].text == ""  # honest empty text back to the pipeline


def test_transcribe_page_fast_mode_reuses_significant_text(monkeypatch) -> None:
    """Fast mode returns significant native text verbatim, no OCR, no vision."""
    from src.extractor import transcribe_page

    ocr_calls: list[Path] = []
    monkeypatch.setattr("src.extractor._local_ocr", lambda p: ocr_calls.append(p) or "OCR")
    vision_calls: list[int] = []
    fake = RunnableLambda(lambda page: vision_calls.append(page.index) or "VISION")

    page = PdfPage(index=0, width_pt=100, height_pt=200, text="Q" * 60)
    got = transcribe_page(
        page, chain=fake, settings=Settings(fast_mode=True), document_name="doc.pdf"
    )
    assert got == "Q" * 60
    assert ocr_calls == []  # significant text -> no OCR, no vision
    assert vision_calls == []


# ---------------------------------------------------------------------------
# 3. Correction loop capped at 0
# ---------------------------------------------------------------------------


def test_format_autodraft_fast_truncates_llm_input_to_budget() -> None:
    seen: dict[str, str] = {}

    def _probe(inputs: dict) -> FileOutput:
        seen["text"] = inputs["document_text"]
        return FileOutput()

    cfg = Settings(fast_mode=True)
    long_text = "This is a long supplier invoice line. " * 500  # ~46k chars
    assert len(long_text) > cfg.fast_max_input_chars

    result = format_autodraft(
        document_text=long_text,
        filename="F.pdf",
        chain=RunnableLambda(_probe),
        master=MasterData(),
        settings=cfg,
    )
    assert len(seen["text"]) == cfg.fast_max_input_chars == 3000
    assert seen["text"] == long_text[:3000]
    assert result.file == "F.pdf"


def test_format_autodraft_precision_passes_full_text() -> None:
    seen: dict[str, str] = {}

    def _probe(inputs: dict) -> FileOutput:
        seen["text"] = inputs["document_text"]
        return FileOutput()

    cfg = Settings(fast_mode=False)
    # Short transcripts pass through untouched in precision mode...
    format_autodraft(
        document_text="short text",
        filename="F.pdf",
        chain=RunnableLambda(_probe),
        master=MasterData(),
        settings=cfg,
    )
    assert seen["text"] == "short text"
    # ...while oversized multi-page transcripts stay capped at
    # MAX_STRUCTURING_CHARS (head+tail kept) so prompt tokens stay under the
    # provider's per-minute budget without dropping later-page totals.
    long_text = "x" * (MAX_STRUCTURING_CHARS + 1000)
    format_autodraft(
        document_text=long_text,
        filename="F.pdf",
        chain=RunnableLambda(_probe),
        master=MasterData(),
        settings=cfg,
    )
    assert len(seen["text"]) <= MAX_STRUCTURING_CHARS
    assert seen["text"].startswith("x")
    assert seen["text"].endswith("x")


# ---------------------------------------------------------------------------
# 4. Compact prompt (fewer input prompt tokens)
# ---------------------------------------------------------------------------


def test_fast_system_prompt_is_substantially_shortened() -> None:
    assert len(_STRUCTURING_SYSTEM_PROMPT_FAST) < len(_STRUCTURING_SYSTEM_PROMPT)
    # Same hard rules survive the trim.
    for must_keep in ("quantity", "item_type", "declined", "GOODS|SERVICE|FREIGHT|TAX"):
        assert must_keep in _STRUCTURING_SYSTEM_PROMPT_FAST


def test_build_fast_chain_selects_compact_system_prompt() -> None:
    cfg = Settings(llm_api_key="test-key", fast_mode=True)
    chain = build_autodraft_chain(fast_mode=True, settings=cfg)
    system = next(m for m in chain.first.messages if isinstance(m, SystemMessage))
    assert system.content == _STRUCTURING_SYSTEM_PROMPT_FAST


def test_build_precision_chain_keeps_full_system_prompt() -> None:
    cfg = Settings(llm_api_key="test-key", fast_mode=False)
    chain = build_autodraft_chain(fast_mode=False, settings=cfg)
    system = next(m for m in chain.first.messages if isinstance(m, SystemMessage))
    assert system.content == _STRUCTURING_SYSTEM_PROMPT
