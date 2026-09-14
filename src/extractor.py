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
import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.llm_retry import describe_error, invoke_with_retry
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


def transcription_cache_path(
    document_name: str,
    page_index: int,
    image_fingerprint: str | int,
    cache_dir: Path,
) -> Path:
    """File path for a page's on-disk transcription cache entry.

    The key is an MD5 of (document name, page number, image fingerprint).
    Callers pass the **content digest** of the rasterised page (see
    :func:`_image_fingerprint`), which — unlike a byte *length* — can never
    collide for two different page images of equal size.
    """
    digest = hashlib.md5(
        f"{document_name}|{page_index}|{image_fingerprint}".encode()
    ).hexdigest()
    return cache_dir / f"{digest}.txt"


def _image_fingerprint(image_path: Path) -> str:
    """Stable content fingerprint (MD5) of a rendered page PNG.

    Content-hashed rather than length-hashed so two different pages of the
    same byte size never share a cache entry.
    """
    try:
        return hashlib.md5(image_path.read_bytes()).hexdigest()
    except OSError:
        return "missing"


def read_transcription(text_path: Path) -> str | None:
    """Return the cached transcription at *text_path*, or ``None`` on a miss.

    A missing, unreadable, empty or undecodable (i.e. partially-written or
    corrupted) cache file is treated as a miss so the page is re-transcribed
    instead of silently reusing a truncated transcription.
    """
    try:
        text = text_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not text.strip():
        return None
    return text


def write_transcription(text_path: Path, text: str) -> None:
    """Persist *text* to *text_path*; a failed write is logged, never fatal.

    Written atomically (temp file + rename) so a hard exit can never leave a
    partially-written cache entry that would be trusted on the next run.
    """
    try:
        text_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = text_path.with_name(text_path.name + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(text_path)
    except OSError as exc:
        log.warning("Could not cache transcription %s: %s", text_path, exc)


def _has_api_key(settings: Settings) -> bool:
    """True when an OpenAI-compatible key is available for the vision path."""
    return bool(settings.provider_api_key())


def _local_ocr(image_path: Path) -> str | None:
    """Attempt local CPU-based OCR via pytesseract.

    Returns the extracted text, or ``None`` if pytesseract is not installed
    or OCR fails. This avoids Vision LLM API calls entirely for scanned
    pages when a usable local OCR engine is available.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip() if text and text.strip() else None
    except Exception as exc:
        log.debug("Local OCR failed for %s: %s", image_path, exc)
        return None


def build_image_transcription_chain(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build the LangChain runnable that transcribes one page image to text.

    Args:
        model: vision-capable model name; defaults to ``INV_VISION_MODEL``.
        temperature: sampling temperature; defaults to the configured value.
        settings: settings source; defaults to the process singleton.

    Returns:
        A LangChain ``Runnable`` whose input is a :class:`PdfPage` and whose
        output is the transcribed text (``str``).

    Raises:
        RuntimeError: when no API key is configured for the model.
    """
    cfg = settings or get_settings()
    model_name = model or cfg.vision_model
    temp = cfg.extraction_temperature if temperature is None else temperature

    try:
        llm: BaseChatModel = ChatOpenAI(
            model=model_name,
            temperature=temp,
            base_url=cfg.provider_base_url(),
            api_key=cfg.provider_api_key(),
            timeout=cfg.llm_timeout,
        )
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
        try:
            response = invoke_with_retry(llm, [human])
        except Exception as exc:  # noqa: BLE001 - surface the raw provider error
            log.error(
                "Vision transcription failed for page %s (model '%s'): %s",
                page.index,
                model_name,
                describe_error(exc),
            )
            raise
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
    document_name: str | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Resolve the full text of a single page.

    Pages with a usable embedded text layer are returned verbatim. Pages that
    are image-only are sent through ``chain``; if no chain is available the
    result is ``""`` and, when ``require_vision`` is set, an error is raised
    because the extractor would otherwise silently drop real content.

    When ``document_name`` is given, transcriptions are persisted to disk
    (``cache_dir``, default the configured transcription cache): a content hash
    of the rendered page decides the file, so a re-run skips the vision API
    entirely for pages already transcribed. Empty or corrupted cache entries
    are treated as misses and re-transcribed.

    Args:
        page: the page to process.
        chain: the vision runnable; ``None`` disables vision transcription.
        require_vision: fail hard when an image page cannot be transcribed;
            defaults to the configured setting.
        document_name: source document file name; enables on-disk caching.
        cache_dir: transcription cache directory; defaults to the configured
            value when ``document_name`` is set.

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

    # Fast mode: attempt local CPU-based OCR before falling back to the
    # vision LLM chain.  This eliminates base64 encoding overhead and
    # Vision TPM rate-limit pressure for scanned pages.
    if cfg.fast_mode and page.image_path is not None:
        ocr_text = _local_ocr(page.image_path)
        if ocr_text:
            log.info("Local OCR extracted %s chars from page %s", len(ocr_text), page.index)
            return ocr_text

    if chain is None:
        message = (
            f"Page {page.index} of an image-only document needs vision "
            "transcription but no vision chain/API key is configured."
        )
        if need:
            raise RuntimeError(message)
        log.warning("Returning empty text for image page %s (%s)", page.index, message)
        return ""

    text_path: Path | None = None
    if document_name:
        cache = cache_dir or cfg.resolved_transcription_dir()
        fingerprint = _image_fingerprint(page.image_path) if page.image_path else ""
        text_path = transcription_cache_path(document_name, page.index, fingerprint, cache)
        cached = read_transcription(text_path)
        if cached is not None:
            log.info(
                "Transcription cache hit for %s page %s (%s)",
                document_name,
                page.index,
                fingerprint,
            )
            return cached

    text = chain.invoke(page)
    if text_path is not None:
        write_transcription(text_path, text)
    log.info("Transcribed page %s (%s chars)", page.index, len(text))
    return text


def extract_document_text(
    document: Document,
    chain: Any | None = None,
    settings: Settings | None = None,
    *,
    cache_dir: Path | None = None,
) -> Document:
    """Extract the full text of a document, page by page.

    Mutates ``document`` in place: image-only pages get their ``text`` field
    populated from the vision chain and their ``image_path`` cleared once
    transcribed (token spend is per page, never repeated). Pages already in the
    on-disk transcription cache are read back without any API call and without
    pacing.

    Args:
        document: the ingested document.
        chain: the vision transcription runnable; built on demand from the
            default model when ``None`` and an API key is present.
        settings: settings source; defaults to the process singleton.
        cache_dir: transcription cache directory; defaults to the configured
            value.

    Returns:
        The same document with every page's text populated where possible.
    """
    cfg = settings or get_settings()
    cache = cache_dir or cfg.resolved_transcription_dir()

    need_chain = any(not p.text.strip() and p.image_path for p in document.pages)
    effective_chain = chain
    if effective_chain is None and need_chain and _has_api_key(cfg):
        effective_chain = _chain_for(cfg.vision_model, cfg.extraction_temperature)
        log.info(
            "Using vision model '%s' for %s image pages",
            cfg.vision_model,
            len(document.needs_vision),
        )

    pacing = cfg.effective_vision_pacing_delay()

    completed: list[PdfPage] = []
    vision_calls = 0
    vision_count = 0
    for page in document.pages:
        if page.image_path is not None:
            text_path = transcription_cache_path(
                document.filename,
                page.index,
                _image_fingerprint(page.image_path),
                cache,
            )
            cached = read_transcription(text_path)
            if cached is not None:
                # Cache hit: reuse the persisted transcription, no API call and
                # no TPM budget consumed, so no pacing is needed either.
                log.info(
                    "Transcription cache hit for %s page %s",
                    document.filename,
                    page.index,
                )
                text = cached
            else:
                # Pacing: sleep between consecutive vision API calls (not before
                # the first, and never for text-layer pages) to keep Groq
                # TPM/RPM flat while transcribing multi-page scans.
                if vision_calls > 0 and pacing > 0:
                    time.sleep(pacing)
                text = transcribe_page(
                    page,
                    effective_chain,
                    require_vision=cfg.require_vision,
                    document_name=document.filename,
                    cache_dir=cache,
                )
                vision_calls += 1
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
