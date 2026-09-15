"""Tests for the fast-testing-mode configuration knobs (:mod:`src.config`)."""

from __future__ import annotations

from src.config import Settings


def test_precision_mode_defaults() -> None:
    cfg = Settings(fast_mode=False)
    assert cfg.fast_mode is False
    # Effective values equal the configured precision-mode knobs.
    assert cfg.effective_max_retries() == cfg.max_retries
    # Structuring falls back to INV_TEXT_MODEL when no explicit override set.
    assert cfg.effective_structuring_model() == cfg.text_model
    assert cfg.effective_vision_pacing_delay() == cfg.vision_pacing_delay
    assert cfg.effective_text_layer_min_chars() == cfg.text_layer_min_chars
    # No input cap in precision mode: the full page text reaches the LLM.
    assert cfg.effective_max_input_chars() == 0
    # Text-token budget in precision mode (4096, NOT the 16K vision cap).
    assert cfg.effective_max_tokens() == cfg.text_max_tokens
    assert cfg.effective_max_tokens() == 4096


def test_fast_mode_short_circuits_retries_pacing_and_vision() -> None:
    cfg = Settings(fast_mode=True, vision_pacing_delay=4.0)
    # Single-pass extraction: the correction loop is capped at 0.
    assert cfg.effective_max_retries() == 0
    # Inter-call pacing is disabled entirely.
    assert cfg.effective_vision_pacing_delay() == 0.0
    # The "more than 50 characters" text-layer rule kicks in.
    assert cfg.effective_text_layer_min_chars() == 50
    # Fast mode swaps to its own lightweight model.
    assert cfg.effective_structuring_model() == cfg.fast_structuring_model
    # The structuring prompt's input text is hard-capped at 3000 chars.
    assert cfg.effective_max_input_chars() == 3000
    # Output tokens hard-capped at 1000 to prevent rambling.
    assert cfg.effective_max_tokens() == 1000


def test_hybrid_provider_defaults() -> None:
    """Vision targets HF Serverless; text reasoning targets Groq."""
    cfg = Settings()
    # Vision offloaded to the Hugging Face router.
    assert cfg.vision_model == "zai-org/GLM-4.5V"
    assert cfg.vision_base_url == "https://router.huggingface.co/v1"
    # Text reasoning follows INV_TEXT_MODEL / INV_TEXT_BASE_URL when set.
    assert cfg.text_model == "meta-llama/Llama-3.3-70B-Instruct"
    assert cfg.text_base_url == "https://router.huggingface.co/v1"
    assert cfg.groq_base_url == "https://api.groq.com/openai/v1"
    assert cfg.structuring_model == ""  # no override -> text_model wins
    assert cfg.fast_structuring_model == "meta-llama/Llama-3.3-70B-Instruct"
    assert cfg.effective_structuring_model() == cfg.text_model


def test_vision_batching_and_blank_skip_defaults() -> None:
    cfg = Settings()
    assert cfg.vision_batch_size == 1  # one page per batch
    assert cfg.vision_batch_pause == 2.0  # 2s between page requests
    assert cfg.vision_skip_blank_pages is True
    # Per-page pacing stays configurable via INV_VISION_PACING_DELAY.
    assert cfg.vision_pacing_delay == 2.0


def test_vision_batching_knobs_are_overridable() -> None:
    cfg = Settings(vision_batch_size=3, vision_batch_pause=1.5, vision_skip_blank_pages=False)
    assert cfg.vision_batch_size == 3
    assert cfg.vision_batch_pause == 1.5
    assert cfg.vision_skip_blank_pages is False


def test_fast_mode_respects_explicit_overrides() -> None:
    cfg = Settings(
        fast_mode=True,
        fast_text_layer_min_chars=60,
        fast_structuring_model="llama-3.3-70b-versatile",
        fast_max_retries=2,
        fast_max_input_chars=3000,
        fast_max_tokens=2048,
    )
    assert cfg.effective_text_layer_min_chars() == 60
    assert cfg.effective_structuring_model() == "llama-3.3-70b-versatile"
    assert cfg.effective_max_retries() == 2
    assert cfg.effective_max_input_chars() == 3000
    assert cfg.effective_max_tokens() == 2048


# ---------------------------------------------------------------------------
# Dual-provider LLM factories
# ---------------------------------------------------------------------------


def test_get_vision_llm_targets_hf_router() -> None:
    cfg = Settings(
        hf_token="hf-aabbcc",
        vision_model="zai-org/GLM-4.5V",
        extraction_temperature=0.1,
    )
    llm = cfg.get_vision_llm()
    assert llm.model_name == "zai-org/GLM-4.5V"
    assert llm.openai_api_base == "https://router.huggingface.co/v1"
    assert llm.openai_api_key.get_secret_value() == "hf-aabbcc"
    assert llm.temperature == 0.1


def test_get_vision_llm_accepts_override_and_raises_without_token(monkeypatch) -> None:
    import pytest

    cfg = Settings(hf_token="hf-key")
    llm = cfg.get_vision_llm(model="my-vision-model", temperature=0.7, max_tokens=2048)
    assert llm.model_name == "my-vision-model"
    assert llm.temperature == 0.7
    assert llm.max_tokens == 2048

    # No HF token AND no legacy OPENAI_API_KEY fallback -> hard error.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        Settings(hf_token="").get_vision_llm()


def test_get_text_llm_targets_groq() -> None:
    cfg = Settings(
        groq_api_key="gsk-text",
        text_base_url="",
        text_model="openai/gpt-oss-120b",
        structuring_temperature=0.2,
    )
    llm = cfg.get_text_llm()
    assert llm.model_name == "openai/gpt-oss-120b"
    assert llm.openai_api_base == "https://api.groq.com/openai/v1"
    assert llm.openai_api_key.get_secret_value() == "gsk-text"
    assert llm.temperature == 0.2
    # Completion cap leaves headroom for a complete payables JSON.
    assert llm.max_tokens == 4096


def test_get_text_llm_override_respects_cap(monkeypatch) -> None:
    import pytest

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Settings(groq_api_key="gsk-text", text_max_tokens=1536)
    llm = cfg.get_text_llm()
    assert llm.max_tokens == 1536
    with pytest.raises(ValueError, match="greater than or equal to"):
        Settings(text_max_tokens=64)


def test_get_text_llm_uses_fast_model_in_fast_mode() -> None:
    cfg = Settings(groq_api_key="gsk-text", fast_mode=True)
    assert cfg.get_text_llm().model_name == cfg.fast_structuring_model


def test_get_text_llm_with_override_and_raises_without_key(monkeypatch) -> None:
    import pytest

    cfg = Settings(groq_api_key="gsk-text", text_base_url="")
    assert cfg.get_text_llm(model="llama-3.3-70b-versatile").model_name == "llama-3.3-70b-versatile"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        Settings(groq_api_key="", text_base_url="").get_text_llm()


def test_legacy_llm_api_key_still_builds_text_llm(monkeypatch) -> None:
    """INV_LLM_API_KEY remains a valid fallback for local test setups."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Settings(
        groq_api_key="",
        text_base_url="",
        llm_api_key="test-key",
        llm_base_url="https://example.com/v1",
    )
    llm = cfg.get_text_llm()
    assert llm.openai_api_key.get_secret_value() == "test-key"
    # The Groq base URL always wins for text reasoning (provider-scoped).
    assert llm.openai_api_base == "https://api.groq.com/openai/v1"
