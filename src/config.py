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

    # ── Hugging Face Serverless (vision transcription) ───────────────────
    #: User Access Token for the Hugging Face Inference API / Serverless
    #: router. Read from ``HF_TOKEN`` (no ``INV_`` prefix — it is the standard
    #: Hugging Face env var name).
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    #: Hugging Face router base URL (OpenAI-compatible endpoint). Provider
    #: selection is automatic (fastest available provider for the model).
    vision_base_url: str = Field(default="https://router.huggingface.co/v1")
    #: Model that transcribes image-only pages. Now a Hugging Face Serverless
    #: Vision model — vision is fully offloaded from Groq, so image payloads no
    #: longer compete for Groq's TPM budget (read from ``INV_VISION_MODEL``).
    vision_model: str = Field(default="zai-org/GLM-4.5V")

    # ── Groq (text reasoning) ────────────────────────────────────────────
    #: Groq API key for text/schema/validation loops (no ``INV_`` prefix — it
    #: is the standard Groq env var name).
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    #: Groq OpenAI-compatible base URL.
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")
    #: Text-reasoning endpoint override. When set, :meth:`get_text_llm` routes
    #: *text* reasoning here instead of ``groq_base_url`` (and uses
    #: ``hf_token`` for the key when no Groq credential is usable), enabling a
    #: pure Full-HuggingFace architecture that is immune to Groq's daily
    #: TPD/TPM quotas. Read from ``INV_TEXT_BASE_URL``.
    text_base_url: str = Field(default="")
    #: Text-reasoning model (document text -> autodraft JSON, validation
    #: corrector). Groq serves the OpenAI gpt-oss weights (read from
    #: ``INV_TEXT_MODEL``).
    text_model: str = Field(default="openai/gpt-oss-120b")

    # ── LLM provider (legacy OpenAI-compatible fallback) ─────────────────
    #: Base URL override for a single OpenAI-compatible endpoint used as a
    #: *fallback* by :meth:`get_vision_llm` / :meth:`get_text_llm` when the
    #: dedicated provider key is absent. Defaults to ``OPENAI_BASE_URL``.
    llm_base_url: str = Field(default="")
    #: Legacy API key fallback; defaults to ``OPENAI_API_KEY``.
    llm_api_key: str = Field(default="")
    #: Output token budget for one structured-response call. Kept generous so
    #: multi-line-item invoices (now with per-line quantity/unit_price) are not
    #: truncated mid-JSON by the provider.
    llm_max_tokens: int = Field(default=16384, ge=256, le=1_000_000)
    #: Output token budget for one *text*-reasoning call (structuring + the
    #: multi-agent validation/corrector loop). Generous enough that a full
    #: payables JSON (supplier, buyer, payment_terms, line items, taxes and
    #: totals) completes without mid-payload truncation: the Hugging Face
    #: Serverless router otherwise stops generation at a low default limit,
    #: clipping the JSON and surfacing as a ``JSONDecodeError``. Read from
    #: ``INV_TEXT_MAX_TOKENS``.
    text_max_tokens: int = Field(default=4096, ge=128, le=1_000_000)
    #: Read timeout (seconds) for a single provider call. A hung connection
    #: becomes ``openai.APITimeoutError`` — which the retry wrapper retries —
    #: instead of freezing the process/UI forever. Kept at 45 s so a benchmark
    #: can fail fast instead of waiting out a stalled serverless cold start.
    llm_timeout: float = Field(default=45.0, ge=1, le=3600)
    #: Total call attempts for one LLM inference (first call + retries).
    #: Default 1 = fail fast during benchmarking: a transient provider error
    #: surfaces immediately instead of silently burning back-off windows.
    llm_retry_attempts: int = Field(default=1, ge=1, le=10)
    #: Maximum single wait (seconds) a retry may sleep before we refuse to
    #: block the pipeline. A Groq 429 with ``Retry-After: 300`` is rarer than a
    #: benchmark hang — anything above this cap raises
    #: :class:`~src.llm_retry.BenchmarkTimeoutException` to fail fast.
    llm_max_wait: float = Field(default=60.0, ge=1, le=600)

    # ── LLM extraction (vision transcription) ───────────────────────────
    #: Sampling temperature for vision transcription (referenced here; the
    #: vision model itself now lives in the Hugging Face section above).
    extraction_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    #: When True, refuse to silently return an empty transcription for an
    #: image-only page if no LLM is configured; raises instead.
    require_vision: bool = Field(default=True)
    #: Seconds to sleep between consecutive vision-transcription calls within
    #: one multi-page document. Pacing keeps TPM/RPM usage flat while a batch
    #: transcribes many scanned pages back-to-back. ``0`` disables it.
    vision_pacing_delay: float = Field(default=2.0, ge=0.0, le=3600.0)
    #: Pages transcribed per vision *batch* in precision mode. Default 1: every
    #: page is its own batch, so ``vision_batch_pause`` (default 2s) sleeps
    #: between every page request, giving Groq's TPM counters a window to reset
    #: on long scanned documents. Pages inside a batch are still transcribed one
    #: at a time, each separated by ``vision_pacing_delay``.
    vision_batch_size: int = Field(default=1, ge=1, le=50)
    #: Extra seconds to pause between vision *batches* (on top of the per-page
    #: pacing). With the default batch size of 1 this becomes the inter-page
    #: delay that keeps Groq TPM/RPM usage flat. ``0`` disables the pause.
    vision_batch_pause: float = Field(default=2.0, ge=0.0, le=3600.0)
    #: When True, near-blank scanned pages are detected (via the rendered
    #: page's pixel histogram) and skipped before any vision call, so blank /
    #: appendix pages never burn vision tokens (see
    #: :func:`src.extractor._is_blank_page`).
    vision_skip_blank_pages: bool = Field(default=True)

    # ── LLM structuring (document text -> autodraft) ────────────────────
    #: Legacy override for the text-reasoning model used by the structuring /
    #: corrector chains. ``""`` (default) delegates to :attr:`text_model`
    #: (``INV_TEXT_MODEL``). Kept so an explicit ``INV_STRUCTURING_MODEL`` still
    #: pins a specific Groq model without editing code.
    structuring_model: str = Field(default="")
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
    #: Lightweight structuring model for fast mode. Groq serves the OpenAI
    #: gpt-oss weights (``openai/gpt-oss-120b``); fast mode keeps the same
    #: well-served model but with the strict token/retry caps below.
    fast_structuring_model: str = Field(default="openai/gpt-oss-120b")
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
        """Text-reasoning model used by the proposer/corrector chains (Groq).

        Fast mode swaps the heavy model for ``fast_structuring_model``; precision
        mode prefers an explicit ``structuring_model`` override and otherwise
        falls back to ``text_model`` (``INV_TEXT_MODEL``).
        """
        if self.fast_mode:
            return self.fast_structuring_model
        return self.structuring_model or self.text_model

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
        1000) to prevent model rambling and minimise latency. Precision mode
        uses the ``text_max_tokens`` budget (default 4096) so a full payables
        JSON is never truncated server-side.
        """
        return self.fast_max_tokens if self.fast_mode else self.text_max_tokens

    # ------------------------------------------------------------------
    def provider_base_url(self) -> str:
        """Resolve the LLM provider base URL (setting, else ``OPENAI_BASE_URL``)."""
        return self.llm_base_url or os.environ.get("OPENAI_BASE_URL", "")

    def provider_api_key(self) -> str:
        """Resolve the LLM provider API key (setting, else ``OPENAI_API_KEY``)."""
        return self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")

    # ------------------------------------------------------------------
    def get_vision_llm(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Build a ``ChatOpenAI`` wired to the Hugging Face Serverless router.

        This is the vision transcription client: image page payloads run on
        Hugging Face's Serverless inference (``vision_base_url`` +
        ``HF_TOKEN``), fully offloading image tokens from Groq's TPM budget.

        Args:
            model: vision model name; defaults to ``INV_VISION_MODEL``.
            temperature: sampling temperature; defaults to the configured value.
            max_tokens: output cap; defaults to ``llm_max_tokens``.

        Returns:
            A configured ``langchain_openai.ChatOpenAI`` instance.

        Raises:
            RuntimeError: when ``HF_TOKEN`` (or the legacy OpenAI-compatible
                key fallback) is not configured.
        """
        from langchain_openai import ChatOpenAI

        api_key = self.hf_token or self.provider_api_key()
        if not api_key:
            raise RuntimeError(
                "Cannot build the vision LLM: no Hugging Face credentials "
                "configured (set HF_TOKEN)."
            )
        return ChatOpenAI(
            model=model or self.vision_model,
            temperature=self.extraction_temperature if temperature is None else temperature,
            base_url=self.vision_base_url,
            api_key=api_key,
            max_tokens=self.llm_max_tokens if max_tokens is None else max_tokens,
            timeout=self.llm_timeout,
        )

    def get_text_llm(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Build a ``ChatOpenAI`` for text reasoning.

        Default route: Groq (``groq_base_url`` + ``GROQ_API_KEY``), which
        serves the OpenAI gpt-oss weights. When ``text_base_url`` is set
        (``INV_TEXT_BASE_URL``) the call goes there instead and, falling back
        to the legacy OpenAI-compatible key, the Hugging Face token is used —
        this is the pure Full-HF configuration that is independent of Groq's
        TPD/TPM quotas.

        Args:
            model: text-reasoning model; defaults to
                :meth:`effective_structuring_model` (``INV_TEXT_MODEL``).
            temperature: sampling temperature; defaults to the configured value.
            max_tokens: output cap; defaults to ``text_max_tokens`` (4096) —
                enough headroom for a complete payables JSON on a serverless
                endpoint that would otherwise default to a low token limit.

        Returns:
            A configured ``langchain_openai.ChatOpenAI`` instance.

        Raises:
            RuntimeError: when no usable credential is configured for the
                selected route.
        """
        from langchain_openai import ChatOpenAI

        if self.text_base_url:
            base_url = self.text_base_url
            api_key = (
                self.hf_token
                or self.groq_api_key
                or self.llm_api_key
                or os.environ.get("OPENAI_API_KEY", "")
            )
            if not api_key:
                raise RuntimeError(
                    "Cannot build the text LLM on the text_base_url route: no "
                    "token configured (set HF_TOKEN or GROQ_API_KEY)."
                )
        else:
            base_url = self.groq_base_url
            api_key = self.groq_api_key or self.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "Cannot build the text LLM: no Groq credentials configured "
                    "(set GROQ_API_KEY or INV_LLM_API_KEY)."
                )
        return ChatOpenAI(
            model=model or self.effective_structuring_model(),
            temperature=self.structuring_temperature if temperature is None else temperature,
            base_url=base_url,
            api_key=api_key,
            max_tokens=(self.effective_max_tokens() if max_tokens is None else max_tokens),
            timeout=self.llm_timeout,
        )

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
