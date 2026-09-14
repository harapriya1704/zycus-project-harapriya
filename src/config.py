"""Runtime configuration for the pipeline.

Settings are read from environment variables (prefix ``INV_``) and an optional
``.env`` file. This keeps all tunable knobs — paths, model names, render DPI —
in one place instead of scattering them through the code.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Root of the repository: the parent of the ``src`` package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Load ``.env`` into the process environment so the OpenAI-compatible client
#: (Groq via ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``) sees the credentials even
#: when they are only declared in the file, not the shell.
load_dotenv(PROJECT_ROOT / ".env")


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
    #: Directory vision transcriptions are cached to, so a re-run does not re-spend
    #: API tokens on pages already transcribed (kept out of the repo).
    transcription_cache_dir: Path = Field(
        default=Path.cwd() / ".cache" / "transcriptions",
        description="Where cached vision transcriptions live.",
    )

    # ── LLM provider (OpenAI-compatible; Groq in production) ─────────────
    #: Base URL for an OpenAI-compatible endpoint. Defaults to the
    #: ``OPENAI_BASE_URL`` environment variable (set to Groq in ``.env``).
    llm_base_url: str = Field(default="")
    #: API key for that endpoint. Defaults to ``OPENAI_API_KEY``.
    llm_api_key: str = Field(default="")
    #: Output token budget for one structured-response call. Kept generous so
    #: multi-line-item invoices (now with per-line quantity/unit_price) are not
    #: truncated mid-JSON by the provider.
    llm_max_tokens: int = Field(default=16384, ge=256, le=1_000_000)
    #: Read timeout (seconds) for a single provider call. A hung connection
    #: becomes ``openai.APITimeoutError`` — which the retry wrapper retries —
    #: instead of freezing the process/UI forever.
    llm_timeout: float = Field(default=300.0, ge=1, le=3600)

    # ── LLM extraction (vision transcription) ───────────────────────────
    #: Model that transcribes image-only pages (must support vision/image_url
    #: input — a text-only model rejects the base64 payload). Read from
    #: ``INV_VISION_MODEL``; defaults to the multimodal gateway model
    #: ``qwen/qwen3.8-27b``.
    vision_model: str = Field(default="qwen/qwen3.8-27b")
    extraction_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    #: When True, refuse to silently return an empty transcription for an
    #: image-only page if no LLM is configured; raises instead.
    require_vision: bool = Field(default=True)
    #: Seconds to sleep between consecutive vision-transcription calls within
    #: one multi-page document. Pacing keeps TPM/RPM usage flat while a batch
    #: transcribes many scanned pages back-to-back. ``0`` disables it.
    vision_pacing_delay: float = Field(default=2.0, ge=0.0, le=3600.0)

    # ── LLM structuring (document text -> autodraft) ────────────────────
    structuring_model: str = Field(default="openai/gpt-oss-120b")
    structuring_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Multi-agent validation (day 2) ───────────────────────────────────
    #: Maximum number of correction iterations in the validation loop; the
    #: loop hard-stops here to guarantee it cannot run forever.
    max_retries: int = Field(default=3, ge=1, le=20)

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    def provider_base_url(self) -> str:
        """Resolve the LLM provider base URL (setting, else ``OPENAI_BASE_URL``)."""
        return self.llm_base_url or os.environ.get("OPENAI_BASE_URL", "")

    def provider_api_key(self) -> str:
        """Resolve the LLM provider API key (setting, else ``OPENAI_API_KEY``)."""
        return self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")

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

    def resolved_transcription_dir(self) -> Path:
        """Absolute vision-transcription cache directory."""
        return self._abs(self.transcription_cache_dir)

    @staticmethod
    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
