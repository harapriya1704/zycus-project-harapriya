"""Document text extraction via a LangChain vision-LLM chain.

This is the "read the document" stage. A document arrives as a
:class:`src.pdf_reader.Document`; for every page whose embedded text layer is
usable we reuse it verbatim (no API spend), and for image-only pages we send the
rasterised page to a vision-capable chat model built through LangChain
(:class:`langchain_openai.ChatOpenAI`).

The extraction model is deliberately kept as a thin, swappable LangChain
``Runnable`` (:func:`build_image_transcription_chain`) so tests can inject a
fake model and later phases can swap OCR vendors without touching callers.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.logging_conf import get_logger
from src.pdf_reader import Document, PdfError, PdfPage

log = get_logger(__name__)

#: Data-URI prefix for PNG bytes sent to the vision model.
_PNG_DATA_URI = "data:image/png;base64,{}"

#: System instruction for the transcription step. Kept short and literal: we
#: only ask for a faithful transcription, not for extraction — extraction and
#: structuring happen in the next stage (:mod:`src.formatter`).
_TRANSCRIPTION_SYSTEM_PROMPT = """You are a precise OCR assistant for supplier documents.

Transcribe the page image EXACTLY as it appears. Rules:
- Reproduce every number exactly as printed (keep thousand separators and locale formats as printed).
- Keep table structure legible: one table row per line, columns separated by ' | '.
- Do not translate, summarise, or correct values.
- Do not invent content. If a region is illegible, write [illegible].
Return only the transcription, with no preamble."""


def _data_uri(image_path: Path) -> str:
    """Base64-encode a PNG as a ``data:image/png;base64,...`` URI."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return _PNG_DATA_URI.format(encoded)


def _has_api_key(_settings: Settings) -> bool:
    """True when an OpenAI-compatible key is available for the vision path."""
    return bool(os.getenv("OPENAI_API_KEY"))


def build_image_transcription_chain(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build the LangChain runnable that transcribes one page image to text.

    Args:
        model: vision-capable model name; defaults to ``INV_EXTRACTION_MODEL``.
        temperature: sampling temperature; defaults to the configured value.
        settings: settings source; defaults to the process singleton.

    Returns:
        A LangChain ``Runnable`` whose input is a :class:`PdfPage` and whose
        output is the transcribed text (``str``).

    Raises:
        RuntimeError: when no API key is configured for the model.
    """
    cfg = settings or get_settings()
    model_name = model or cfg.extraction_model
    temp = cfg.extraction_temperature if temperature is None else temperature

    try:
        llm: BaseChatModel = ChatOpenAI(model=model_name, temperature=temp)
    except Exception as exc:  # pragma: no cover - env-dependent
        log.error("Could not construct ChatOpenAI: %s", exc)
        raise

    def _transcribe(page: PdfPage) -> str:
        if page.image_path is None:
            raise PdfError(f"Page {page.index} has no rasterised image to transcribe")
        human = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": _data_uri(page.image_path)}},
                {"type": "text", "text": f"Transcribe page {page.index + 1} of this document."},
            ]
        )
        response = llm.invoke([human])
        return str(response.content).strip()

    chain = RunnableLambda(_transcribe, name="image_transcription_chain")
    chain = chain.with_config({"run_name": "image_transcription_chain"})
    return chain


@lru_cache(maxsize=4)
def _chain_for(model: str, temperature: float) -> Any:
    """Cached transcription chain per (model, temperature)."""
    return build_image_transcription_chain(model=model, temperature=temperature)


def transcribe_page(
    page: PdfPage,
    chain: Any | None = None,
    *,
    require_vision: bool | None = None,
) -> str:
    """Resolve the full text of a single page.

    Pages with a usable embedded text layer are returned verbatim. Pages that
    are image-only are sent through ``chain``; if no chain is available the
    result is ``""`` and, when ``require_vision`` is set, an error is raised
    because the extractor would otherwise silently drop real content.

    Args:
        page: the page to process.
        chain: the vision runnable; ``None`` disables vision transcription.
        require_vision: fail hard when an image page cannot be transcribed;
            defaults to the configured setting.

    Returns:
        The page's text (possibly empty).

    Raises:
        RuntimeError: when an image-only page cannot be transcribed and vision
            is required.
    """
    if page.text.strip():
        return page.text

    cfg = get_settings()
    need = cfg.require_vision if require_vision is None else require_vision

    if chain is None:
        message = (
            f"Page {page.index} of an image-only document needs vision "
            "transcription but no vision chain/API key is configured."
        )
        if need:
            raise RuntimeError(message)
        log.warning("Returning empty text for image page %s (%s)", page.index, message)
        return ""

    text = chain.invoke(page)
    log.info("Transcribed page %s (%s chars)", page.index, len(text))
    return text


def extract_document_text(
    document: Document,
    chain: Any | None = None,
    settings: Settings | None = None,
) -> Document:
    """Extract the full text of a document, page by page.

    Mutates ``document`` in place: image-only pages get their ``text`` field
    populated from the vision chain and their ``image_path`` cleared once
    transcribed (token spend is per page, never repeated).

    Args:
        document: the ingested document.
        chain: the vision transcription runnable; built on demand from the
            default model when ``None`` and an API key is present.
        settings: settings source; defaults to the process singleton.

    Returns:
        The same document with every page's text populated where possible.
    """
    cfg = settings or get_settings()

    need_chain = any(not p.text.strip() and p.image_path for p in document.pages)
    effective_chain = chain
    if effective_chain is None and need_chain and _has_api_key(cfg):
        effective_chain = _chain_for(cfg.extraction_model, cfg.extraction_temperature)
        log.info(
            "Using vision model '%s' for %s image pages",
            cfg.extraction_model,
            len(document.needs_vision),
        )

    completed: list[PdfPage] = []
    vision_count = 0
    for page in document.pages:
        if page.image_path is not None:
            text = transcribe_page(page, effective_chain, require_vision=cfg.require_vision)
            if text:
                vision_count += 1
            page = PdfPage(
                index=page.index,
                width_pt=page.width_pt,
                height_pt=page.height_pt,
                text=text,
            )
        completed.append(page)

    document.pages = completed
    log.info(
        "Extracted %s chars from %s (%s pages, %s via vision)",
        len(document.text()),
        document.filename,
        len(document.pages),
        vision_count,
    )
    return document


def _cli() -> int:
    """Minimal CLI: ``python -m src.extractor [subdir]``.

    Extracts text for every PDF under the given directory (default:
    ``documents/``) and prints per-file character counts — a smoke test for the
    extraction stage without writing output files.
    """
    import sys

    from src.config import get_settings
    from src.logging_conf import configure_logging
    from src.pdf_reader import read_pdf

    cfg = get_settings()
    configure_logging(level=cfg.log_level, log_dir=cfg.resolved_output_dir() / "logs")

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.resolved_documents_dir()
    if target.is_file():
        targets: list[Path] = [target]
    else:
        targets = sorted(target.glob("*.pdf"))

    if not targets:
        log.error("No PDFs found under %s", target)
        return 1

    cumulative = 0
    for pdf in targets:
        try:
            doc = extract_document_text(read_pdf(pdf, dpi=cfg.pdf_render_dpi))
            chars = len(doc.text())
            cumulative += chars
            log.info("%s: %s chars", doc.filename, chars)
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            log.error("%s: FAILED (%s)", pdf.name, exc)
    log.info("Total extracted chars: %s", cumulative)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
