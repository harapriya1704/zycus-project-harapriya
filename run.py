"""One-command runner for the day-1 pipeline.

    python run.py [--input-dir DIR] [--output-dir DIR]

Processes every ``*.pdf`` in ``documents/`` (default) and writes
``output/<name>.json`` for each — the output contract the grader consumes.

Prerequisites: ``pip install -r requirements.txt`` and an ``OPENAI_API_KEY``
in the environment (or ``.env``) for document text with an embedded layer of
fewer than ``INV_TEXT_LAYER_MIN_CHARS`` characters; text-layer documents work
without a key, but still require one for structured-output generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_settings
from src.logging_conf import configure_logging
from src.pipeline import process_directory

logger_ready = False


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
    parser.add_argument("--log-level", default=None, help="Log level (default: INV_LOG_LEVEL or INFO)")
    parser.add_argument("--list", action="store_true", help="Only list matching PDFs, then exit")
    args = parser.parse_args(argv)

    cfg = get_settings()
    documents_dir = (args.input_dir or cfg.resolved_documents_dir()).resolve()
    output_dir = (args.output_dir or cfg.resolved_output_dir()).resolve()
    _configure_logging(args.log_level or cfg.log_level, output_dir)

    if args.list:
        for pdf in sorted(documents_dir.glob("*.pdf")):
            print(pdf.name)
        return 0

    summary = process_directory(documents_dir, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["files_processed"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
