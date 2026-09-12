"""Centralised logging for the pipeline.

Every module obtains its logger through :func:`get_logger`; output goes to
stdout (and a rotating file under ``output/logs/`` when the output dir
exists). The root level is configured once from :class:`src.config.Settings`.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

#: User-facing logger namespace that all modules inherit from.
PACKAGE_LOGGER = "invoice_pipeline"

_configuration_already: bool = False


def _configure(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the package logger exactly once per process."""
    global _configuration_already
    if _configuration_already:
        return
    _configuration_already = True

    root = logging.getLogger(PACKAGE_LOGGER)
    root.setLevel(level.upper())
    root.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "invoice_pipeline.log",
                maxBytes=5_000_000,
                backupCount=2,
                encoding="utf-8",
            )
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem edge case
            root.warning("Could not attach file log handler: %s", exc)

    sys.excepthook = _log_unhandled


def _log_unhandled(exc_type, exc_value, exc_tb) -> None:  # pragma: no cover
    logging.getLogger(PACKAGE_LOGGER).critical(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
    )


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Public entry point; safe to call multiple times (idempotent)."""
    _configure(level=level, log_dir=log_dir)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the package namespace.

    Args:
        name: Module name; ``__name__`` is the expected argument.

    Returns:
        A configured :class:`logging.Logger`.
    """
    configure_logging()
    return logging.getLogger(f"{PACKAGE_LOGGER}.{name}")
