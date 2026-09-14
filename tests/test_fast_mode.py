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
    build_autodraft_chain,
    format_autodraft,
)
from src.master_data import MasterData
from src.pdf_reader import Document, PdfPage, read_pdf
from src.schemas import FileOutput


def _text_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(40, 40, page.rect.width - 40, page.rect.height - 40), text
    )
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
    fast_doc = read_pdf(
        pdf, min_text_chars=50, dpi=72, render_dir=tmp_path / "fast_render"
    )
    assert fast_doc.pages[0].image_path is None  # text layer reused, no PNG
    assert fast_doc.pages[0].text.strip()

    precise_doc = read_pdf(
        pdf, min_text_chars=80, dpi=72, render_dir=tmp_path / "precise_render"
    )
    assert precise_doc.pages[0].image_path is not None  # same page rasterised


def test_extract_fast_mode_disables_pacing(tmp_path: Path, monkeypatch) -> None:
    """Even a 9s pacing delay becomes 0.0 in fast mode."""
    sleeps: list[float] = []
    monkeypatch.setattr("src.extractor.time.sleep", lambda s: sleeps.append(s))
    fake = RunnableLambda(lambda page: f"OCR {page.index}")

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
    assert sleeps == []
    assert [p.text for p in doc.pages] == ["OCR 0", "OCR 1"]


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
    long_text = "x" * 9000
    format_autodraft(
        document_text=long_text,
        filename="F.pdf",
        chain=RunnableLambda(_probe),
        master=MasterData(),
        settings=cfg,
    )
    assert seen["text"] == long_text  # no budget cap outside fast mode


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
    system = next(
        m for m in chain.first.messages if isinstance(m, SystemMessage)
    )
    assert system.content == _STRUCTURING_SYSTEM_PROMPT_FAST


def test_build_precision_chain_keeps_full_system_prompt() -> None:
    cfg = Settings(llm_api_key="test-key", fast_mode=False)
    chain = build_autodraft_chain(fast_mode=False, settings=cfg)
    system = next(
        m for m in chain.first.messages if isinstance(m, SystemMessage)
    )
    assert system.content == _STRUCTURING_SYSTEM_PROMPT
