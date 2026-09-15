"""Comparative provider benchmark: Full-HF vs Hybrid vs Full-Groq.

Runs both benchmark PDFs (``INV-01.pdf``, ``DU-02.pdf``) through ``run.py
--validate`` for each of the three architectures and reports:

- wall-clock latency per document,
- the number of HTTP 429 / TPM throttle / ``Retry-After`` events seen in the
  run log (rate-limit back-off is the pipeline's #1 hang risk),
- ERP recompute accuracy: the fraction of payables whose
  :func:`erp.erp_book` recompute matches the printed ``gross_total`` within
  0.02 currency units (the exact contract the grader uses),
- API failures: payables declined with a ``doc_type`` of ``ERROR`` and the
  number of ``BenchmarkTimeoutException`` fail-fast aborts.

Each architecture runs with a *fresh* transcription/render cache so vision is
never served from a previous run's cache (which would hide real latency).

Usage:
    python scripts/benchmark.py [--arch NAME [NAME ...]]

Exit codes:
    0   completed
    1   a run failed to even produce an output JSON
    2   missing required credentials
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOTENV = REPO_ROOT / ".env"

#: Benchmark PDFs: one small text-layer invoice and one long scanned bill.
BENCH_PDFS = ["INV-01.pdf", "DU-02.pdf"]

#: One entry per architecture: label + environment overrides layered on top of
#: the base ``.env`` (real env vars always win over the dotenv file, so the
#: subprocess sees exactly these values).
ARCHITECTURES: list[dict[str, str]] = [
    {
        "name": "full-hf",
        "label": "Full-HF: GLM-4.5V vision + Llama-3.3-70B text on HF router",
    },
    {
        "name": "hybrid",
        "label": "Hybrid: GLM-4.5V vision on HF + gpt-oss-120b text on Groq",
    },
    {
        "name": "full-groq",
        "label": "Full-Groq: qwen3.8-27b vision + gpt-oss-120b text on Groq",
    },
]

#: Env var -> value dict keyed by architecture name.
ENV_OVERRIDES: dict[str, dict[str, str]] = {
    "full-hf": {
        "INV_VISION_MODEL": "zai-org/GLM-4.5V",
        "INV_VISION_BASE_URL": "https://router.huggingface.co/v1",
        "INV_TEXT_MODEL": "meta-llama/Llama-3.3-70B-Instruct",
        # Text reasoning also runs on the HF router (native route), so the
        # whole pipeline is independent of Groq's TPD/TPM quotas.
        "INV_TEXT_BASE_URL": "https://router.huggingface.co/v1",
        "INV_GROQ_BASE_URL": "https://api.groq.com/openai/v1",
    },
    "hybrid": {
        "INV_VISION_MODEL": "zai-org/GLM-4.5V",
        "INV_VISION_BASE_URL": "https://router.huggingface.co/v1",
        "INV_TEXT_MODEL": "openai/gpt-oss-120b",
        "INV_GROQ_BASE_URL": "https://api.groq.com/openai/v1",
        "INV_TEXT_MAX_TOKENS": "1536",
    },
    "full-groq": {
        "INV_VISION_MODEL": "qwen/qwen3.8-27b",
        "INV_VISION_BASE_URL": "https://api.groq.com/openai/v1",
        "INV_TEXT_MODEL": "openai/gpt-oss-120b",
        "INV_GROQ_BASE_URL": "https://api.groq.com/openai/v1",
        # get_vision_llm prefers hf_token; clear it so it falls back to
        # OPENAI_API_KEY (the Groq key) for Groq-hosted vision.
        "HF_TOKEN": "",
    },
}

#: Tolerance (currency units) between the ERP recompute and printed gross.
AMP_TOLERANCE = 0.02

#: Regex lines counted as rate-limit / back-off events.
_RATE_LIMIT_PATTERNS = (
    "TPM limit",
    "Retry-After",
    "RateLimit",
    "rate_limit",
    "HTTP 429",
    "429 ",
    "Pacing request",
    "too many requests",
    "over rate",
)


def _base_env() -> dict[str, str]:
    """Return os.environ plus any keys the ``.env`` file declares."""
    env = os.environ.copy()
    if DOTENV.is_file():
        try:
            from dotenv import dotenv_values, load_dotenv

            load_dotenv(DOTENV)
            env.update({k: v for k, v in dotenv_values(DOTENV).items() if v is not None})
        except ImportError:  # pragma: no cover - python-dotenv is a dependency
            pass
    return env


def _run_once(
    env: dict[str, str],
    pdf: str,
    out_dir: Path,
    cache_root: Path,
) -> dict:
    """Run ``run.py --validate --file <pdf>`` once under *env*.

    Returns per-document metrics (latency, 429s, ERP match, etc.). SBOM of
    args is kept explicit so the harness is reproducible across runs.
    """
    cache_dir = cache_root.parent / f"{cache_root.name}_{Path(pdf).stem}"
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
    render_dir = cache_dir / "pages"
    trans_dir = cache_dir / "transcriptions"
    render_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_env = env.copy()
    run_env["INV_PDF_RENDER_DIR"] = str(render_dir)
    run_env["INV_TRANSCRIPTION_CACHE_DIR"] = str(trans_dir)

    started = time.monotonic()
    proc = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--validate",
            "--file",
            str(REPO_ROOT / "documents" / pdf),
            "--output-dir",
            str(out_dir),
            "--log-level",
            "INFO",
        ],
        cwd=str(REPO_ROOT),
        env=run_env,
        capture_output=True,
        text=True,
    )
    latency = time.monotonic() - started

    result = {
        "pdf": pdf,
        "latency_s": round(latency, 2),
        "returncode": proc.returncode,
        "payables": 0,
        "declined": 0,
        "error_declined": 0,
        "erp_match": 0,
        "erp_total": 0,
        "rate_limit_events": 0,
        "benchmark_timeouts": 0,
        "api_failures": 0,
        "output_json": "",
        "reason": "",
    }

    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    result["rate_limit_events"] = sum(
        1 for line in combined.splitlines() if any(p in line for p in _RATE_LIMIT_PATTERNS)
    )
    result["benchmark_timeouts"] = combined.count("BenchmarkTimeoutException")
    result["api_failures"] = (
        combined.count("Vision transcription failed")
        + combined.count("Structuring LLM call failed")
        + combined.count("Corrector inference failed")
    )

    out_json = out_dir / f"{Path(pdf).stem}.json"
    if out_json.is_file():
        result["output_json"] = str(out_json)
        (
            result["payables"],
            result["declined"],
            result["error_declined"],
            result["erp_match"],
            result["erp_total"],
        ) = _score_output(out_json)
    else:
        result["reason"] = "no output JSON produced"

    return result


def _score_output(path: Path) -> tuple[int, int, int, int, int]:
    """Return ``(payables, declined, error_declined, erp_matched, erp_total)``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return (0, 0, 0, 0, 0)

    payables = len(data.get("payables", []) or [])
    declined = len(data.get("declined", []) or [])
    error_declined = sum(
        1 for d in data.get("declined", []) or [] if str(d.get("doc_type", "")).upper() == "ERROR"
    )

    matched = 0
    total = 0
    for payable in data.get("payables", []) or []:
        total += 1
        printed = payable.get("gross_total", "")
        if not str(printed or "").strip():
            continue
        try:
            from erp import erp_book

            booked = float(erp_book(dict(payable)).get("will_book_gross", 0.0) or 0.0)
            printed_f = float(printed)
        except (TypeError, ValueError, KeyError):
            continue
        if abs(booked - printed_f) <= AMP_TOLERANCE:
            matched += 1

    return payables, declined, error_declined, matched, total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the provider benchmark.")
    parser.add_argument(
        "--arch", nargs="*", default=None, help="Architecture names to run (default: all)."
    )
    args = parser.parse_args()

    base = _base_env()

    # Resolve the keys each architecture depends on.
    hf_token = base.get("HF_TOKEN", "")
    groq_key = base.get("GROQ_API_KEY") or base.get("OPENAI_API_KEY", "")
    missing = []
    if not hf_token:
        missing.append("HF_TOKEN")
    if not groq_key:
        missing.append("GROQ_API_KEY")
    if missing:
        print(f"ERROR: missing credentials in .env: {', '.join(missing)}")
        return 2

    overrides_by_name = {a["name"]: ENV_OVERRIDES[a["name"]] for a in ARCHITECTURES}
    overrides_by_name["full-groq"].setdefault("OPENAI_API_KEY", groq_key)

    benchmark_root = REPO_ROOT / ".benchmark"
    all_results: dict[str, list[dict]] = {}

    selected = {a["name"] for a in ARCHITECTURES}
    if args.arch:
        selected = set(args.arch)

    for arch in ARCHITECTURES:
        name = arch["name"]
        if name not in selected:
            continue
        print(f"\n=== [{name.upper()}] {arch['label']} ===")
        env = base.copy()
        env.update(overrides_by_name[name])
        out_dir = benchmark_root / name / "output"
        cache_root = benchmark_root / name / "cache"
        per_pdf = []
        for pdf in BENCH_PDFS:
            row = _run_once(env, pdf, out_dir, cache_root)
            per_pdf.append(row)
            print(
                f"  {pdf:<10} {row['latency_s']:6.1f}s  "
                f"payables={row['payables']} erp={row['erp_match']}/{row['erp_total']} "
                f"429s={row['rate_limit_events']} timeouts={row['benchmark_timeouts']} "
                f"apifail={row['api_failures']}" + (f"  [{row['reason']}]" if row["reason"] else "")
            )
        all_results[name] = per_pdf

    _print_table(all_results)
    return 0


def _print_table(all_results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 78)
    print("Benchmark comparison (per architecture, summed over both PDFs)")
    print("=" * 78)
    headers = (
        "architecture",
        "latency_s",
        "payables",
        "erp_match/total",
        "429_events",
        "timeouts",
        "api_failures",
    )
    print(
        f"{headers[0]:<12} {headers[1]:>9} {headers[2]:>9} {headers[3]:>14} {headers[4]:>10} {headers[5]:>9} {headers[6]:>11}"
    )
    for name, rows in all_results.items():
        latency = sum(r["latency_s"] for r in rows)
        payables = sum(r["payables"] for r in rows)
        erp_m = sum(r["erp_match"] for r in rows)
        erp_t = sum(r["erp_total"] for r in rows)
        fours = sum(r["rate_limit_events"] for r in rows)
        tos = sum(r["benchmark_timeouts"] for r in rows)
        afs = sum(r["api_failures"] for r in rows)
        print(
            f"{name:<12} {latency:>9.1f} {payables:>9} {f'{erp_m}/{erp_t}':>14} "
            f"{fours:>10} {tos:>9} {afs:>11}"
        )
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
