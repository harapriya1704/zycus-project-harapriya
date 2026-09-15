"""Pre-flight model availability check for the hybrid/baseline benchmarks.

Probes every candidate model across Hugging Face Serverless (router) and Groq
against an OpenAI-compatible ``/chat/completions`` endpoint, then prints which
models are usable as VISION and TEXT backends for this pipeline.

Usage:
    python scripts/check_models.py

Exit codes:
    0   at least one VISION and one TEXT model are available
    1   no usable configuration (all candidates failed)
    2   could not load credentials from the environment/.env
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV = REPO_ROOT / ".env"

#: Endpoints the pipeline actually targets (OpenAI-compatible chat completions).
HF_ROUTER = "https://router.huggingface.co/v1"
GROQ = "https://api.groq.com/openai/v1"


def _red_png_uri() -> str:
    """A small but valid 64x64 red PNG, base64-encoded as a data-URI.

    A real image is required: some VLMs (e.g. zai-org/GLM-4.5V) reject sub-16px
    test images with a 400 'invalid image' error even though they are fine on
    real scanned pages, which would give false NOT_DEPLOYED verdicts.
    """
    try:
        from PIL import Image

        img = Image.new("RGB", (64, 64), (200, 30, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    except ImportError:  # pragma: no cover - PIL is a project dependency
        encoded = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP"
            "4z8AAAASBAMAAAD1Y7lrAAAAAElFTkSuQmCC"
        )
    return f"data:image/png;base64,{encoded}"

#: Roles we need to fill. ``text`` candidates are probed with a text-only
#: payload; ``vision`` candidates are probed with an image + text payload.
#: ``preferred`` marks the documented primary vs. the fail-over alternative.
CANDIDATES: list[dict[str, Any]] = [
    # -- Hugging Face Serverless (router.huggingface.co/v1) --------------
    {"provider": "hf", "role": "vision", "name": "Qwen/Qwen2.5-VL-7B-Instruct", "base_url": HF_ROUTER, "preferred": "PRIMARY"},
    {"provider": "hf", "role": "vision", "name": "zai-org/GLM-4.5V", "base_url": HF_ROUTER, "preferred": "ALTERNATE"},
    {"provider": "hf", "role": "text", "name": "meta-llama/Llama-3.3-70B-Instruct", "base_url": HF_ROUTER, "preferred": "PRIMARY"},
    {"provider": "hf", "role": "text", "name": "openai/gpt-oss-120b", "base_url": HF_ROUTER, "preferred": "ALTERNATE"},
    # -- Groq ------------------------------------------------------------
    {"provider": "groq", "role": "text", "name": "llama-3.3-70b-versatile", "base_url": GROQ, "preferred": "PRIMARY"},
    {"provider": "groq", "role": "text", "name": "openai/gpt-oss-120b", "base_url": GROQ, "preferred": "ALTERNATE"},
    {"provider": "groq", "role": "vision", "name": "qwen/qwen3.8-27b", "base_url": GROQ, "preferred": "ALTERNATE"},
]

#: How long one probe request may take before we call the endpoint unhealthy.
PROBE_TIMEOUT = 45.0


def _env() -> dict[str, str]:
    """Load credentials (``.env`` if present) and the keys we need."""
    if DOTENV.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(DOTENV)
        except ImportError:
            pass
    if not os.environ.get("HF_TOKEN"):
        print("ERROR: HF_TOKEN not found in environment or .env")
        sys.exit(2)
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not found in environment or .env")
        sys.exit(2)
    return {"hf": os.environ["HF_TOKEN"], "groq": os.environ["GROQ_API_KEY"]}


def _payload(role: str) -> dict[str, Any]:
    if role == "vision":
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _red_png_uri()}},
            {"type": "text", "text": "Reply with the word OK."},
        ]
    else:
        content = "Reply with the word OK."
    return {"model": "", "messages": [{"role": "user", "content": content}], "max_tokens": 8}


def probe(base_url: str, api_key: str, model: str, role: str) -> tuple[str, float]:
    """Return ``(status_tag, latency_s)`` for one model probe.

    Stats:
        AVAILABLE        HTTP 200 with usable text in the reply.
        NOT_DEPLOYED     HTTP 404 / 400 'model_not_supported' -> model not on
                         this account/provider -> switch to alternative.
        AUTH             HTTP 401 (token invalid).
        NO_VISION        HTTP 400 about image handling (text-only model).
        COLD_START_503   503; waits one warm-up window (5 s) and retries once.
        RATE_LIMITED     HTTP 429 even for a tiny probe.
        UNREACHABLE      connection/timeout (host down, TLS, etc.).
    """
    payload = _payload(role)
    payload["model"] = model
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=PROBE_TIMEOUT) as client:
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
            elapsed = time.monotonic() - t0
            if resp.status_code == 200:
                try:
                    text = resp.json()["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    text = ""
                return ("AVAILABLE" if text.strip() else "AVAILABLE_EMPTY"), elapsed
            if resp.status_code == 401:
                return "AUTH", elapsed
            if resp.status_code == 429:
                return "RATE_LIMITED", elapsed
            if resp.status_code == 503 and attempt == 1:
                time.sleep(5.0)  # serverless cold-loader; one warm-up window
                continue
            body = resp.text[:300]
            if resp.status_code == 404:
                return "NOT_DEPLOYED", elapsed
            if resp.status_code == 400:
                if "model_not_supported" in body or "does not exist" in body.lower():
                    return "NOT_DEPLOYED", elapsed
                return "NO_VISION", elapsed
            return f"HTTP_{resp.status_code}", elapsed
        except (httpx.TimeoutException, httpx.ConnectError, httpx.TransportError):
            elapsed = time.monotonic() - t0
            if attempt == 1:
                continue
            return "UNREACHABLE", elapsed
    return "COLD_START_503", time.monotonic() - t0  # pragma: no cover


def _fmt(rows: list[dict[str, Any]]) -> None:
    for r in rows:
        ok = r["tag"] in {"AVAILABLE", "AVAILABLE_EMPTY"}
        print(
            f"{'OK ' if ok else '-- '}[{r['tag']:<15}] {r['latency']:5.1f}s "
            f"{r['preferred']:<9} {r['provider']:>4} {r['role']:<6} {r['name']}"
        )


def main() -> int:
    keys = _env()
    results: list[dict[str, Any]] = []
    for c in CANDIDATES:
        tag, latency = probe(c["base_url"], keys[c["provider"]], c["name"], c["role"])
        results.append({**c, "tag": tag, "latency": latency})
    _fmt(results)

    def _pick(provider: str, role: str) -> str:
        for r in results:
            if r["provider"] == provider and r["role"] == role and r["tag"] in {"AVAILABLE", "AVAILABLE_EMPTY"}:
                return r["name"]
        return "NONE"

    hf_vision = _pick("hf", "vision")
    hf_text = _pick("hf", "text")
    groq_text = _pick("groq", "text")
    groq_vision = _pick("groq", "vision")

    print("\n-- Verdicts ----------------------------------------------")
    print(f"  Hybrid:           HF vision = {hf_vision:<45} | Groq text = {groq_text}")
    print(f"  Full HuggingFace: HF vision = {hf_vision:<45} | HF text    = {hf_text}")
    print(f"  Full Groq:        Groq vision = {groq_vision:<45} | Groq text  = {groq_text}")

    if hf_vision != "NONE" and (groq_text != "NONE" or hf_text != "NONE"):
        print("\nRESULT: at least one viable configuration found.")
        return 0
    print("\nRESULT: no viable configuration (all candidates unavailable).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
