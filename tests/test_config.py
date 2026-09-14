"""Tests for the fast-testing-mode configuration knobs (:mod:`src.config`)."""

from __future__ import annotations

from src.config import Settings


def test_precision_mode_defaults() -> None:
    cfg = Settings(fast_mode=False)
    assert cfg.fast_mode is False
    # Effective values equal the configured precision-mode knobs.
    assert cfg.effective_max_retries() == cfg.max_retries
    assert cfg.effective_structuring_model() == cfg.structuring_model
    assert cfg.effective_vision_pacing_delay() == cfg.vision_pacing_delay
    assert cfg.effective_text_layer_min_chars() == cfg.text_layer_min_chars
    # No input cap in precision mode: the full page text reaches the LLM.
    assert cfg.effective_max_input_chars() == 0
    # Full token budget in precision mode.
    assert cfg.effective_max_tokens() == cfg.llm_max_tokens


def test_fast_mode_short_circuits_retries_pacing_and_vision() -> None:
    cfg = Settings(fast_mode=True, vision_pacing_delay=4.0)
    # Single-pass extraction: the correction loop is capped at 0.
    assert cfg.effective_max_retries() == 0
    # Inter-call pacing is disabled entirely.
    assert cfg.effective_vision_pacing_delay() == 0.0
    # The "more than 50 characters" text-layer rule kicks in.
    assert cfg.effective_text_layer_min_chars() == 50
    # The heavy structuring model is swapped for the lightweight fast one.
    assert cfg.effective_structuring_model() == "llama-3.1-8b-instant"
    # The structuring prompt's input text is hard-capped at 3000 chars.
    assert cfg.effective_max_input_chars() == 3000
    # Output tokens hard-capped at 1000 to prevent rambling.
    assert cfg.effective_max_tokens() == 1000


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
