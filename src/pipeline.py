"""End-to-end per-file pipeline and directory run.

Wires the stages together for one PDF:

    read (src.pdf_reader) -> extract (src.extractor) -> structure (src.formatter)

and writes per-file ``output/X.json`` documents matching the output contract
in ``AUTODRAFT_SCHEMA.md``. Every failure degrades to an honest ``declined``
entry; the run never dies on one bad file.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from langchain_core.runnables import Runnable

from src.agents import ValidationLoop
from src.config import Settings, get_settings
from src.extractor import extract_document_text
from src.formatter import format_autodraft
from src.llm_retry import describe_error
from src.logging_conf import get_logger
from src.master_data import MasterData
from src.pdf_reader import read_pdf
from src.schemas import Declined, FileOutput

log = get_logger(__name__)


def _decline(error: Exception, filename: str) -> FileOutput:
    """Wrap any stage failure as an honest declined entry."""
    return FileOutput(
        file=filename,
        declined=[Declined(doc_type="ERROR", reason=str(error) or error.__class__.__name__)],
    )


def process_document(
    pdf_path: Path,
    *,
    settings: Settings | None = None,
    vision_chain: Runnable | None = None,
    structuring_chain: Runnable | None = None,
    master: MasterData | None = None,
    validation_loop: ValidationLoop | None = None,
) -> FileOutput:
    """Process a single PDF into a :class:`FileOutput`.

    Args:
        pdf_path: the source PDF.
        settings: settings source; defaults to the process singleton.
        vision_chain: override for the vision transcription runnable (tests).
        structuring_chain: override for the structuring runnable (tests).
        master: override for the master data (tests).
        validation_loop: when provided, routes the document through the
            proposer/validator/corrector loop (day-2 agents) instead of the
            single-shot structuring chain. The loop inherits ``structuring_chain``
            as its proposer when not set explicitly on the loop.

    Returns:
        A validated per-file result; the ``file`` field is the PDF's name.
    """
    cfg = settings or get_settings()
    document_text = ""
    try:
        document = read_pdf(
            pdf_path,
            min_text_chars=cfg.effective_text_layer_min_chars(),
            dpi=cfg.pdf_render_dpi,
        )
        # Vision routing: the vision model (qwen/qwen3.8-27b) is invoked ONLY
        # for pages rasterised as images (embedded text layer shorter than the
        # effective ``text_layer_min_chars`` threshold ==> 0 usable chars).
        # Text-layer pages — including every page above the fast-mode "50 char"
        # rule — are returned verbatim by ``extract_document_text`` and never
        # reach the vision API, so scanning a text-layer PDF costs zero vision
        # tokens.
        extract_document_text(document, chain=vision_chain, settings=cfg)
        document_text = document.text()
    except Exception as exc:  # noqa: BLE001 - degrade to argued 'declined'
        log.error("%s: extraction failed: %s", pdf_path.name, describe_error(exc))
        return _decline(exc, pdf_path.name)

    try:
        if validation_loop is not None:
            result = validation_loop.run(document_text, pdf_path.name).file_output
        else:
            result = format_autodraft(
                document_text,
                filename=pdf_path.name,
                chain=structuring_chain,
                master=master,
                settings=cfg,
            )
    except Exception as exc:  # noqa: BLE001 - e.g. no credentials for text pages
        log.error("%s: structuring failed: %s", pdf_path.name, describe_error(exc))
        return _decline(exc, pdf_path.name)

    log.info(
        "%s -> %s payable(s), %s declined",
        pdf_path.name,
        len(result.payables),
        len(result.declined),
    )
    return result


def write_output(result: FileOutput, output_dir: Path) -> Path:
    """Serialise a per-file result to ``output_dir/<file>.json``.

    If the source file name is a directory path (unlikely), the basename is
    used; the output always ends in ``.json``.

    Args:
        result: the result to write.
        output_dir: destination directory (created if absent).

    Returns:
        The path that was written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target = Path(result.file).name
    target = Path(target).stem + ".json"
    out_path = output_dir / target
    out_path.write_text(result.to_json() + "\n", encoding="utf-8")
    log.debug("Wrote %s (%s bytes)", out_path, out_path.stat().st_size)
    return out_path


def process_directory(
    documents_dir: Path,
    output_dir: Path,
    *,
    settings: Settings | None = None,
    vision_chain: Runnable | None = None,
    structuring_chain: Runnable | None = None,
    master: MasterData | None = None,
    validation_loop: ValidationLoop | None = None,
) -> dict[str, int | Counter]:
    """Process every ``*.pdf`` under ``documents_dir``, writing results.

    Args:
        documents_dir: folder of input PDFs.
        output_dir: destination for per-file JSON.
        settings: settings source.
        vision_chain / structuring_chain / master: stage overrides (tests).
        validation_loop: day-2 agent loop; when provided, every document goes
            through propose -> validate -> correct instead of single-shot
            structuring.

    Returns:
        A summary: file counts, payable counts, declined reason tally.
    """
    cfg = settings or get_settings()
    pdfs = sorted(documents_dir.glob("*.pdf"))
    if not pdfs:
        log.warning("No PDFs found under %s", documents_dir)

    master_effective = master or MasterData.load_default(settings=cfg)
    completed = 0
    payable_count = 0
    declined_reasons: Counter = Counter()
    for pdf in pdfs:
        try:
            result = process_document(
                pdf,
                settings=cfg,
                vision_chain=vision_chain,
                structuring_chain=structuring_chain,
                master=master_effective,
                validation_loop=validation_loop,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            # Unrecoverable per-document failure (e.g. an API error that survives
            # every retry): degrade to an honest 'declined' entry, log it, and let
            # the batch continue with the remaining PDFs.
            log.error(
                "%s: unhandled processing error, marking declined and continuing: %s",
                pdf.name,
                describe_error(exc),
            )
            result = _decline(exc, pdf.name)
        try:
            write_output(result, output_dir)
        except Exception as exc:  # noqa: BLE001 - a disk failure must not kill the batch
            log.error("%s: output write failed (%s); continuing", pdf.name, describe_error(exc))
        payable_count += len(result.payables)
        for entry in result.declined:
            declined_reasons[entry.doc_type if entry.doc_type else "DECLINED"] += 1
        completed += 1

    summary = {
        "files_processed": completed,
        "payables": payable_count,
        "declined": declined_reasons,
    }
    log.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary  # type: ignore[return-value]
