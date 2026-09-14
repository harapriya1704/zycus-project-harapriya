"""One-command runner for the day-1 pipeline.

    python run.py [--input-dir DIR] [--output-dir DIR] [--validate] [--fast]

Processes every ``*.pdf`` in ``documents/`` (default) and writes
``output/<name>.json`` for each — the output contract the grader consumes.
With ``--validate``, each document is routed through the multi-agent
validation loop (proposer/validator/corrector, capped at ``INV_MAX_RETRIES``
corrections, default 3) instead of the single-shot structuring chain.
``--fast`` enables Fast Testing Mode: pages with a usable text layer skip the
vision LLM, the correction loop is capped to a single pass and structuring
uses a lightweight model with a hard input-token budget.

Prerequisites: ``pip install -r requirements.txt`` and an ``OPENAI_API_KEY``
(plus ``OPENAI_BASE_URL`` for a Groq-compatible endpoint) in the environment
or ``.env``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.agents import ValidationLoop
from src.config import Settings
from src.logging_conf import configure_logging, get_logger
from src.master_data import MasterData
from src.pipeline import process_directory, process_document, write_output

log = get_logger(__name__)
logger_ready = False

#: Warning logged exactly once when Fast Testing Mode is active.
FAST_MODE_WARNING = (
    "[FAST TESTING MODE ACTIVE] Bypassing Vision API | "
    "Retries capped to 0 | Input text truncated."
)


def _configure_logging(level: str, output_dir: Path) -> None:
    global logger_ready
    if logger_ready:
        return
    configure_logging(level=level, log_dir=output_dir / "logs")
    logger_ready = True


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Turn supplier PDFs into structured autodrafts (one JSON per PDF).",
    )
    parser.add_argument("--input-dir", type=Path, default=None, help="Folder of PDFs (default: <repo>/documents)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder (default: <repo>/output)")
    parser.add_argument("--file", type=Path, default=None, help="Process a single PDF instead of an input folder")
    parser.add_argument("--log-level", default=None, help="Log level (default: INV_LOG_LEVEL or INFO)")
    parser.add_argument("--validate", action="store_true",
                        help="Route documents through the multi-agent validation loop")
    parser.add_argument("--fast", action="store_true",
                        help="Fast Testing Mode: skip vision for text pages, cap retries to 0, fast LLM routing")
    parser.add_argument("--list", action="store_true", help="Only list matching PDFs, then exit")
    args = parser.parse_args(argv)

    cfg = Settings(fast_mode=args.fast)
    documents_dir = (args.input_dir or cfg.resolved_documents_dir()).resolve()
    output_dir = (args.output_dir or cfg.resolved_output_dir()).resolve()
    _configure_logging(args.log_level or cfg.log_level, output_dir)

    if args.fast:
        # Warn once up-front that --fast trades extraction precision for speed.
        log.warning(FAST_MODE_WARNING)

    if args.list:
        for pdf in sorted(documents_dir.glob("*.pdf")):
            print(pdf.name)
        return 0

    loop = None
    if args.validate:
        loop = ValidationLoop(master=MasterData.load_default(settings=cfg), settings=cfg)

    if args.file is not None:
        pdf = args.file.resolve()
        if not pdf.is_file():
            log.error("File not found: %s", pdf)
            return 1
        master = MasterData.load_default(settings=cfg)
        result = process_document(
            pdf,
            settings=cfg,
            master=master,
            validation_loop=loop,
        )
        write_output(result, output_dir)
        declined_reasons: Counter = Counter()
        for entry in result.declined:
            declined_reasons[entry.doc_type if entry.doc_type else "DECLINED"] += 1
        summary = {
            "file": result.file,
            "files_processed": 1,
            "payables": len(result.payables),
            "declined": declined_reasons,
        }
    else:
        summary = process_directory(documents_dir, output_dir, validation_loop=loop)
    if loop is not None:
        summary["validation"] = loop.stats
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["files_processed"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
