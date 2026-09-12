"""PDF ingestion.

Splits a source PDF into pages and, for each page, decides whether its embedded
text layer is usable or whether the page must be rasterised and transcribed by
the vision path.

Responsibilities:
- open a PDF with PyMuPDF and iterate its pages,
- extract the embedded text layer,
- rasterise image-only pages to PNG files (see ``pdf_render_dir``).

This module is deliberately free of any LLM dependency — the extraction stage
(:mod:`src.extractor`) consumes the :class:`Document` / :class:`PdfPage`
values produced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.logging_conf import get_logger

log = get_logger(__name__)


class PdfError(RuntimeError):
    """Raised when a PDF cannot be ingested (corrupt file, bad path, ...)."""


@dataclass(frozen=True)
class PdfPage:
    """One page of a source document.

    Attributes:
        index: 0-based page position within the PDF.
        width_pt / height_pt: page dimensions in PDF points.
        text: embedded text-layer content; ``""`` when the page carries no
            meaningful text (i.e. it is a scanned image).
        image_path: path of the rasterised PNG when ``text`` is unusable;
            ``None`` otherwise. Written lazily on first access by a caller
            that needs vision input.
    """

    index: int
    width_pt: float
    height_pt: float
    text: str = ""
    image_path: Path | None = None


@dataclass
class Document:
    """A source PDF plus its (possibly partially transcribed) pages.

    Attributes:
        path: absolute path of the source PDF.
        pages: per-page content. A page with usable embedded text has
            ``text != ""`` and ``image_path is None``; an image page has
            ``text == ""`` and an ``image_path`` pointing at its PNG.
    """

    path: Path
    pages: list[PdfPage] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Base file name (e.g. ``INV-01.pdf``)."""
        return self.path.name

    @property
    def has_text_layer(self) -> bool:
        """True when at least one page carries usable embedded text."""
        return any(p.text for p in self.pages)

    @property
    def needs_vision(self) -> list[PdfPage]:
        """The pages that must be transcribed by the vision path."""
        return [p for p in self.pages if not p.text.strip()]

    def text(self) -> str:
        """Concatenated embedded text across pages (may be empty)."""
        return "\n\n".join(p.text for p in self.pages if p.text).strip()


def _extract_text_layer(raw_page: pymupdf.Page) -> str:
    """Extract and lightly normalise the embedded text of a page."""
    text = raw_page.get_text("text").strip()
    return text


def _rasterise(doc_stem: str, raw_page: pymupdf.Page, dpi: int, render_dir: Path) -> Path:
    """Render a page to a PNG, returning its file path."""
    render_dir.mkdir(parents=True, exist_ok=True)
    # The pixmap's resolution is governed by the matrix scaling factor.
    zoom = dpi / 72.0
    pix = raw_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    image_path = render_dir / f"{doc_stem}_p{raw_page.number + 1:03d}.png"
    pix.save(image_path)
    log.debug("Rendered page %s -> %s", raw_page.number, image_path)
    return image_path


def read_pdf(
    path: Path | str,
    *,
    min_text_chars: int = 80,
    dpi: int = 180,
    render_dir: Path | None = None,
) -> Document:
    """Ingest a PDF into a :class:`Document`.

    Pages whose embedded text layer has fewer than ``min_text_chars`` non-space
    characters are rasterised (``PdfPage.image_path`` set) instead of being
    transcribed from their text layer.

    Args:
        path: PDF file to open.
        min_text_chars: treat a page as image-only when its text layer is
            shorter than this.
        dpi: rendering resolution for image-only pages.
        render_dir: where page PNGs are written; defaults to the configured
            render cache directory.

    Returns:
        The ingested :class:`Document`.

    Raises:
        PdfError: when the file does not exist or cannot be parsed.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise PdfError(f"PDF not found: {pdf_path}")

    render_target = render_dir
    pages: list[PdfPage] = []
    try:
        with pymupdf.open(pdf_path) as doc:
            for index, raw_page in enumerate(doc):
                text = _extract_text_layer(raw_page)
                if _significant(text, min_text_chars):
                    pages.append(
                        PdfPage(
                            index=index,
                            width_pt=raw_page.rect.width,
                            height_pt=raw_page.rect.height,
                            text=text,
                        )
                    )
                else:
                    if render_target is None:
                        from src.config import get_settings

                        render_target = get_settings().resolved_render_dir()
                    image_path = _rasterise(pdf_path.stem, raw_page, dpi, render_target)
                    pages.append(
                        PdfPage(
                            index=index,
                            width_pt=raw_page.rect.width,
                            height_pt=raw_page.rect.height,
                            image_path=image_path,
                        )
                    )
    except Exception as exc:
        raise PdfError(f"Failed to parse {pdf_path}: {exc}") from exc

    return Document(path=pdf_path, pages=pages)


def _significant(text: str, min_chars: int) -> bool:
    """True when ``text`` has at least ``min_chars`` printable characters."""
    return len("".join(text.split())) >= min_chars
