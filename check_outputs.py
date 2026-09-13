"""check_outputs.py — feed every validated output through the ERP oracle.

Bridges the day-2 validated autodrafts to the grading ERP (:mod:`erp`):

    python check_outputs.py                     # all payables in output/
    python check_outputs.py output/INV-31.json  # a single file

For each payable it prints what the ERP will book (its recompute of the raw
components) alongside the gross the document printed. A MISMATCH here means
the autodraft's components do not recompute to the document's gross — exactly
what the validation agent reports as an ERROR issue during processing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from erp import erp_book
from src.schemas import Autodraft

_DEFAULT_DIR = Path("output")


def _num(v: object) -> float | None:
    if not str(v or "").strip():
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _check_one(payable: dict, path: Path, index: int) -> tuple[bool, str]:
    draft = Autodraft.model_validate(payable)
    booked = erp_book(draft.model_dump(mode="json"))
    printed = _num(payable.get("gross_total"))

    head = f"{path.name}[{index}] ({draft.invoice_number or 'no-number'})"
    if printed is None:
        return True, f"{head}: ERP books {booked}; no printed gross to compare"
    delta = abs(booked["will_book_gross"] - printed)
    status = "MATCH" if delta < 0.01 else "MISMATCH"
    return delta < 0.01, f"{head}: ERP {booked['will_book_gross']:.2f} vs printed {printed:.2f} -> {status}"


def main(argv: list[str] | None = None) -> int:
    targets = []
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        targets = sorted(_DEFAULT_DIR.glob("*.json"))
    else:
        for raw in args:
            p = Path(raw)
            targets += sorted(p.glob("*.json")) if p.is_dir() else [p]

    if not targets:
        print("No output JSON files found. Run `python run.py [--validate]` first.", file=sys.stderr)
        return 1

    payables = 0
    mismatches = 0
    for path in targets:
        blob = json.loads(path.read_text(encoding="utf-8"))
        for index, payable in enumerate(blob.get("payables", [])):
            payables += 1
            ok, line = _check_one(payable, path, index)
            if not ok:
                mismatches += 1
            print(f"  {line}")
    print(f"checked {payables} payables, {mismatches} amount mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
