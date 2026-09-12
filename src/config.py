"""Runtime configuration for the pipeline.

Settings are read from environment variables (prefix ``INV_``) and an optional
``.env`` file. This keeps all tunable knobs — paths, model names, render DPI —
in one place instead of scattering them through the code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Root of the repository: the parent of the ``src`` package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Pipeline settings, overridable via ``INV_*`` env vars / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="INV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────
    documents_dir: Path = Field(default=Path("documents"))
    output_dir: Path = Field(default=Path("output"))
    master_data_dir: Path = Field(default=Path("master_data"))

    # ── Document handling ───────────────────────────────────────────────
    #: Pages whose embedded text layer is shorter than this are considered
    #: image-only and routed to the vision/LLM transcription path.
    text_layer_min_chars: int = Field(default=80, ge=0)
    #: Resolution used when rasterising an image-only page for vision input.
    pdf_render_dpi: int = Field(default=180, ge=72)
    #: Directory rendered page PNGs are written to (kept out of the repo).
    pdf_render_dir: Path = Field(
        default=Path.cwd() / ".cache" / "pages",
        description="Where rasterised pages are cached.",
    )

    # ── LLM extraction (vision transcription) ───────────────────────────
    extraction_model: str = Field(default="gpt-4o")
    extraction_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    #: When True, refuse to silently return an empty transcription for an
    #: image-only page if no LLM is configured; raises instead.
    require_vision: bool = Field(default=True)

    # ── LLM structuring (document text -> autodraft) ────────────────────
    structuring_model: str = Field(default="gpt-4o-mini")
    structuring_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    def resolved_documents_dir(self) -> Path:
        """Absolute documents directory (relative to the repo root)."""
        return self._abs(self.documents_dir)

    def resolved_output_dir(self) -> Path:
        """Absolute output directory (relative to the repo root)."""
        return self._abs(self.output_dir)

    def resolved_master_data_dir(self) -> Path:
        """Absolute master-data directory (relative to the repo root)."""
        return self._abs(self.master_data_dir)

    def resolved_render_dir(self) -> Path:
        """Absolute page-raster cache directory."""
        return self._abs(self.pdf_render_dir)

    @staticmethod
    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
