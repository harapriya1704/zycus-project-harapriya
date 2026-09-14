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

    # ── Fast testing mode (--fast) ───────────────────────────────────────
    #: Master switch for fast local testing: skips the vision LLM for every
    #: page with a usable text layer, disables inter-call pacing, caps the
    #: multi-agent loop to a single pass and routes structuring to a fast,
    #: lightweight model. Accepted trade-off: slightly lower extraction
    #: precision on text-heavy pages.
    fast_mode: bool = Field(default=False)
    #: In fast mode, pages whose embedded text layer carries at least this
    #: many printable characters are reused verbatim — never rasterised to a
    #: PNG, never sent to the vision model (the "> 50 chars" rule).
    fast_text_layer_min_chars: int = Field(default=50, ge=1)
    #: Lightweight structuring model for fast mode (text-only; no images).
    fast_structuring_model: str = Field(default="llama-3.1-8b-instant")
    #: Correction iterations allowed in fast mode. ``0`` = single-pass
    #: extraction: propose + validate, no LLM Corrector iterations.
    fast_max_retries: int = Field(default=0, ge=0, le=20)
    #: Cap (chars) on the raw page text sent to the structuring LLM in fast
    #: mode. ``0`` disables the cap (normal precision mode passes everything).
    fast_max_input_chars: int = Field(default=3000, ge=0)
    #: Hard-cap on output tokens for structuring LLM calls in fast mode.
    #: Prevents model rambling and keeps latency/usage minimal.
    fast_max_tokens: int = Field(default=1000, ge=256, le=1_000_000)

    # ── Logging ─────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------
    def effective_text_layer_min_chars(self) -> int:
        """Text-layer threshold used by the reader.

        Fast mode lowers it to ``fast_text_layer_min_chars`` (50) so pages
        carrying "more than 50 characters" of text skip rasterisation and the
        vision LLM entirely.
        """
        return self.fast_text_layer_min_chars if self.fast_mode else self.text_layer_min_chars

    def effective_structuring_model(self) -> str:
        """Structuring model used by the proposer/corrector chains.

        Fast mode swaps the heavy model for ``fast_structuring_model``
        (e.g. ``llama-3.1-8b-instant`` on Groq).
        """
        return self.fast_structuring_model if self.fast_mode else self.structuring_model

    def effective_vision_pacing_delay(self) -> float:
        """Seconds to sleep between consecutive vision calls (0.0 = disabled).

        Fast mode always disables pacing — local runs want speed, not flat
        provider TPM/RPM.
        """
        return 0.0 if self.fast_mode else self.vision_pacing_delay

    def effective_max_retries(self) -> int:
        """Correction iterations allowed by the validation loop.

        Fast mode forces ``0`` (single-pass): the Corrector Agent is never run
        and only deterministic post-processing resolves the draft.
        """
        return self.fast_max_retries if self.fast_mode else self.max_retries

    def effective_max_input_chars(self) -> int:
        """Cap on the input text sent to the structuring LLM (0 = unlimited).

        Fast mode truncates the raw page text to ``fast_max_input_chars``
        (default 3000 chars per page) so prompt tokens drop sharply on long
        documents.
        """
        return self.fast_max_input_chars if self.fast_mode else 0

    def effective_max_tokens(self) -> int:
        """Output token budget for one structuring/correction LLM call.

        Fast mode hard-caps completion tokens to ``fast_max_tokens`` (default
        1000) to prevent model rambling and minimise latency.
        """
        return self.fast_max_tokens if self.fast_mode else self.llm_max_tokens

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
