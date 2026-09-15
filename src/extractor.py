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
import math
import shutil
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from src.config import Settings, get_settings
from src.llm_retry import describe_error, invoke_with_retry
from src.logging_conf import get_logger
from src.pdf_reader import Document, PdfError, PdfPage

log = get_logger(__name__)

#: Data-URI prefix for JPEG bytes sent to the vision model.
_JPEG_DATA_URI = "data:image/jpeg;base64,{}"
#: Fallback prefix when PIL is unavailable and raw PNG bytes are sent.
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


#: Longest allowed side (px) of a page image sent to the vision model.
_IMAGE_MAX_EDGE = 1000
#: JPEG quality used when compressing page images for vision payloads.
_IMAGE_JPEG_QUALITY = 70


def _data_uri(image_path: Path) -> str:
    """Compress a page image to JPEG and base64-encode it as a data-URI.

    Full-resolution PNG renders dwarf Groq's free TPM budget on multi-page
    scans, forcing 300s ``Retry-After`` back-offs. Before encoding:
    - the image is downscaled so its longest side is at most 1000px
      (``PIL`` ``thumbnail``, LANCZOS — keeps text legible for the vision
      model while slashing payload bytes), and
    - it is recompressed to JPEG at ``quality=70`` (~4-5x smaller than the
      raw PNG).

    When Pillow is unavailable / decoding fails, the raw PNG is sent as-is so
    transcription never breaks.
    """
    try:
        import io

        from PIL import Image
    except ImportError:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return _PNG_DATA_URI.format(encoded)
    try:
        with Image.open(image_path) as img:
            img.thumbnail((_IMAGE_MAX_EDGE, _IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_IMAGE_JPEG_QUALITY)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return _JPEG_DATA_URI.format(encoded)
    except Exception:
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
    digest = hashlib.md5(f"{document_name}|{page_index}|{image_fingerprint}".encode()).hexdigest()
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
    """True when Hugging Face (or the legacy OpenAI-compatible) key is set.

    The vision path is served by the Hugging Face Serverless router; the
    legacy ``provider_api_key`` fallback keeps local test setups working.
    """
    return bool(settings.hf_token or settings.provider_api_key())


#: A local OCR result must carry at least this many printable characters to
#: be trusted; anything shorter is treated as "no readable text" and the page
#: falls through to the Vision API.
_OCR_MIN_CHARS = 50


def _local_ocr(image_path: Path) -> str | None:
    """Attempt local CPU-based OCR via pytesseract.

    The page is downscaled (longest side at most :data:`_IMAGE_MAX_EDGE` px)
    before OCR so the CPU-bound tesseract job stays fast; the extracted text
    is returned only when it carries at least :data:`_OCR_MIN_CHARS`
    printable characters, else ``None``. Returns ``None`` when pytesseract is
    not installed or OCR fails. When usable, the text is handed straight to
    the structuring model — the Vision API call (and its TPM cost) is skipped.
    """
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError:
        return None
    try:
        # The UB-Mannheim installer does not add tesseract.exe to PATH, so
        # point pytesseract at the binary explicitly when it is not on PATH.
        exe = pytesseract.pytesseract.tesseract_cmd or "tesseract"
        if not _tesseract_available(exe):
            for candidate in _TESSERACT_CANDIDATES:
                if _tesseract_available(str(candidate)):
                    pytesseract.pytesseract.tesseract_cmd = str(candidate)
                    break
            else:
                return None
        with Image.open(image_path) as img:
            img.thumbnail((_IMAGE_MAX_EDGE, _IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
            if img.mode not in ("L", "RGB"):
                img = ImageOps.grayscale(img)
            text = pytesseract.image_to_string(img)
        text = text.strip() if text else ""
        if len("".join(text.split())) < _OCR_MIN_CHARS:
            return None
        return text
    except Exception as exc:
        log.debug("Local OCR failed for %s: %s", image_path, exc)
        return None


#: Common install locations for the Tesseract binary on Windows.
_TESSERACT_CANDIDATES: tuple[Path, ...] = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tesseract.exe",
)


def _tesseract_available(exe: str | Path) -> bool:
    """True when ``exe`` names an on-disk tesseract executable."""
    if Path(str(exe)).exists() and Path(str(exe)).is_file():
        return True
    return exe == "tesseract" and shutil.which("tesseract") is not None


#: Fraction of dark pixels below which a rendered page is treated as blank.
_BLANK_INK_THRESHOLD = 0.0005
#: Grayscale cutoff below which a pixel counts as "ink" (0 = black, 255 = white).
_BLANK_INK_CUTOFF = 160


def _is_blank_page(image_path: Path) -> bool:
    """True when the rendered page is effectively blank (no meaningful ink).

    Reads the rasterised PNG's grayscale histogram; when fewer than
    ``_BLANK_INK_THRESHOLD`` (0.05%) of the pixels are dark
    (``<_BLANK_INK_CUTOFF``), the page is treated as blank and its vision call
    (and TPM cost) is skipped. Degrades to ``False`` on any decode failure or
    missing Pillow — real content is never dropped.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            total = gray.width * gray.height
            if total <= 0:
                return False
            hist = gray.histogram()
            dark = sum(hist[:_BLANK_INK_CUTOFF])
            return (dark / total) < _BLANK_INK_THRESHOLD
    except Exception:
        return False


def build_image_transcription_chain(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Any:
    """Build the LangChain runnable that transcribes one page image to text.

    The underlying LLM is created by :meth:`Settings.get_vision_llm` — a
    ``ChatOpenAI`` wired to the Hugging Face Serverless router (``HF_TOKEN``),
    so image transcription never consumes Groq TPM. The chain stays a thin,
    swappable ``Runnable`` so tests inject fakes.

    Args:
        model: vision-capable model name; defaults to ``INV_VISION_MODEL``
            (``zai-org/GLM-4.5V`` on Hugging Face Serverless).
        temperature: sampling temperature; defaults to the configured value.
        settings: settings source; defaults to the process singleton.

    Returns:
        A LangChain ``Runnable`` whose input is a :class:`PdfPage` and whose
        output is the transcribed text (``str``).

    Raises:
        RuntimeError: when no Hugging Face token is configured for the model.
    """
    cfg = settings or get_settings()
    model_name = model or cfg.vision_model
    temp = cfg.extraction_temperature if temperature is None else temperature

    llm: BaseChatModel = cfg.get_vision_llm(model=model_name, temperature=temp)

    def _transcribe(page: PdfPage) -> str:
        if page.image_path is None:
            raise PdfError(f"Page {page.index} has no rasterised image to transcribe")
        human = HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": _data_uri(page.image_path)}},
                {"type": "text", "text": f"Transcribe page {page.index + 1} of this document."},
            ]
        )
        _start = time.perf_counter()
        try:
            response = invoke_with_retry(
                llm,
                [human],
                max_attempts=cfg.llm_retry_attempts,
                max_wait=cfg.llm_max_wait,
            )
        except Exception as exc:  # noqa: BLE001 - surface the raw provider error
            log.error(
                "Vision transcription failed for page %s (model '%s'): %s",
                page.index,
                model_name,
                describe_error(exc),
            )
            raise
        log.info(
            "[Vision HF] Transcribed page %s in %.2fs (model %s)",
            page.index + 1,
            time.perf_counter() - _start,
            model_name,
        )
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
    settings: Settings | None = None,
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
        settings: settings source; defaults to the process singleton. Passed
            down so the fast-mode flag (and related knobs) honoured by the
            caller are respected inside this function too.
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
    cfg = settings or get_settings()

    # ── Fast Testing Mode: HARD Vision API BYPASS ──────────────────────
    # While ``--fast`` is active the Vision LLM chain (e.g.
    # ``qwen/qwen3.8-27b``) must NEVER be invoked, under any circumstances.
    # Native text already extracted via PyMuPDF is reused when significant
    # (>= 50 chars); otherwise a short/empty text layer is supplemented by
    # local CPU-based pytesseract OCR. Whatever that yields — even ``""`` —
    # is returned directly; execution never falls through to the vision chain.
    if cfg.fast_mode:
        text = page.text or ""
        if (
            len("".join(text.split())) < cfg.effective_text_layer_min_chars()
            and page.image_path is not None
        ):
            ocr_text = _local_ocr(page.image_path)
            if ocr_text:
                log.info("Local OCR extracted %s chars from page %s", len(ocr_text), page.index)
                text = ocr_text
        log.info("Fast mode active: Skipped Vision LLM call for page %s.", page.index)
        return text

    if page.text.strip():
        return page.text

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

    # The Vision LLM chain is only ever built for precision mode. Fast mode
    # hard-bypasses it: ``transcribe_page`` resolves image pages via local
    # OCR, so constructing a vision chain here would be pure overhead (and a
    # latent fallback path). Callers may still pass an explicit chain for
    # tests, but it is never invoked in fast mode.
    need_chain = any(not p.text.strip() and p.image_path for p in document.pages)
    effective_chain = chain
    if not cfg.fast_mode and effective_chain is None and need_chain and _has_api_key(cfg):
        effective_chain = _chain_for(cfg.vision_model, cfg.extraction_temperature)
        log.info(
            "Using vision model '%s' for %s image pages",
            cfg.vision_model,
            len(document.needs_vision),
        )

    pacing = cfg.effective_vision_pacing_delay()
    batch_size = cfg.vision_batch_size
    batch_pause = cfg.vision_batch_pause if not cfg.fast_mode else 0.0
    skip_blank = cfg.vision_skip_blank_pages and not cfg.fast_mode

    completed: list[PdfPage] = []
    vision_calls = 0
    vision_count = 0
    ocr_count = 0
    cache_count = 0
    skipped_blank = 0
    batch_calls = 0
    batch_serial = 0
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
                cache_count += 1
            else:
                # Blank pages are detected *before* any pacing sleep so they
                # cost neither a vision call nor a waiting pause. This keeps
                # vision tokens (and TPM) from being spent on blank/appendix
                # pages of multi-page scans.
                if skip_blank and _is_blank_page(page.image_path):
                    log.info(
                        "Skipping near-blank page %s of %s — no vision tokens spent",
                        page.index,
                        document.filename,
                    )
                    text = ""
                    skipped_blank += 1
                else:
                    # Local OCR first: a readable scanned page is transcribed
                    # on the CPU and fed to structuring directly, so no base64
                    # payload ever reaches the Vision model (zero TPM spend).
                    # Only pages OCR cannot read fall through to the batched,
                    # serialised Vision API path below.
                    ocr_text = _local_ocr(page.image_path)
                    if ocr_text and len("".join(ocr_text.split())) >= _OCR_MIN_CHARS:
                        text = ocr_text
                        if text_path is not None:
                            write_transcription(text_path, text)
                        log.info(
                            "Local OCR fallback: %s chars from page %s of %s — Vision API call skipped",
                            len(text),
                            page.index,
                            document.filename,
                        )
                        ocr_count += 1
                    else:
                        # Batched, serialised vision processing: every call is
                        # separated by ``pacing`` seconds (never sent
                        # concurrently), and an extra ``batch_pause`` is added
                        # at batch boundaries (every ``batch_size`` pages) to
                        # stay cleanly under the Groq TPM/RPM ceiling.
                        if vision_calls > 0 and batch_calls == 0 and batch_pause > 0:
                            time.sleep(batch_pause)
                        if vision_calls > 0 and pacing > 0:
                            time.sleep(pacing)
                        batch_serial += 1
                        batch_no = math.ceil(batch_serial / batch_size)
                        log.info(
                            "Vision page %s of %s (batch %s, size %s): transcribing via %s",
                            page.index + 1,
                            document.filename,
                            batch_no,
                            batch_size,
                            cfg.vision_model,
                        )
                        text = transcribe_page(
                            page,
                            effective_chain,
                            settings=cfg,
                            require_vision=cfg.require_vision,
                            document_name=document.filename,
                            cache_dir=cache,
                        )
                        vision_calls += 1
                        batch_calls += 1
                        if batch_calls >= batch_size:
                            batch_calls = 0
                        if text:
                            vision_count += 1
            page = PdfPage(
                index=page.index,
                width_pt=page.width_pt,
                height_pt=page.height_pt,
                text=text,
            )
        completed.append(page)
        log.info(
            "Page %s/%s of %s extracted",
            len(completed),
            len(document.pages),
            document.filename,
        )

    document.pages = completed
    if cfg.fast_mode:
        log.info(
            "Extracted %s chars from %s (%s pages, 0 via vision, %s via local OCR, %s via cache)",
            len(document.text()),
            document.filename,
            len(document.pages),
            ocr_count,
            cache_count,
        )
    else:
        log.info(
            "Extracted %s chars from %s (%s pages, %s via vision, %s via local OCR, %s via cache, %s blank skipped)",
            len(document.text()),
            document.filename,
            len(document.pages),
            vision_count,
            ocr_count,
            cache_count,
            skipped_blank,
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
