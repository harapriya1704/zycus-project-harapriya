"""Integration: validated autodrafts are consumable by the grading ERP.

The whole point of day-2 is that what we emit is byte-compatible with what
``erp.erp_book`` books. These tests pin the contract with the *real* oracle,
not a mock: whatever the validator accepts as consistent today, the ERP must
round-trip to the printed gross.
"""

from __future__ import annotations

import json
from pathlib import Path

from erp import erp_book
from src.config import PROJECT_ROOT, get_settings
from src.master_data import MasterData
from src.schemas import Autodraft
from src.validation import MasterDataValidator

MASTER_DIR = PROJECT_ROOT / "master_data"
_SAMPLE = PROJECT_ROOT / "sample_autodraft.json"


def test_sample_autodraft_erp_roundtrip() -> None:
    """The shipped sample binds to the oracle cleanly (example_check.py)."""
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "example_check.py"), str(_SAMPLE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "will_book_gross = 438.0 EUR" in proc.stdout


def test_validated_payload_is_erp_consumable() -> None:
    """A validator-accepted payable must be directly feedable to erp_book."""
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)

    payable = Autodraft(
        invoice_number="265345",
        currency="EUR",
        gross_total="292.00",
        supplier={"name": "Phocus Direct Communication GmbH", "supplier_id": "2845695"},
        line_items=[
            {
                "description": "Projektmanagement",
                "item_type": "SERVICE",
                "quantity": "4",
                "unit_price": "73.00",
                "total": "292.00",
            }
        ],
    )

    payload = payable.model_dump(mode="json")
    assert validator.oracle(payload) == {"will_book_gross": 292.0, "currency": "EUR"}


def test_sample_autodraft_json_shape_consumable() -> None:
    """sample_autodraft.json (the grader contract example) parses + books."""
    blob = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    payable = Autodraft.model_validate(blob)
    booked = erp_book(payable.model_dump(mode="json"))
    assert booked == {"will_book_gross": 438.0, "currency": "EUR"}


def test_check_outputs_bridge_runs_clean(tmp_path: Path) -> None:
    """check_outputs.py reports MATCH and exits 0 on consistent payables."""

    from erp import erp_book

    payable = Autodraft(
        invoice_number="1", currency="EUR", gross_total="100.00",
        line_items=[{"description": "x", "item_type": "GOODS", "quantity": "2", "unit_price": "50.00", "total": "100.00"}],
    )
    assert erp_book(payable.model_dump(mode="json"))["will_book_gross"] == 100.0

    out = tmp_path / "out"
    out.mkdir()
    (out / "ok.json").write_text(
        payable.model_dump_json(), encoding="utf-8"
    )

    import check_outputs

    args = [str(out)]
    code = check_outputs.main(args)
    assert code == 0


def test_settings_provider_defaults() -> None:
    """Default model targets the Groq-class llama model; retries capped at 3."""
    cfg = get_settings()
    assert cfg.structuring_model == "llama-3.3-70b-versatile"
    assert cfg.max_retries == 3
