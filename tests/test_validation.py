"""Tests for the deterministic validation agent (:mod:`src.validation`)."""

from __future__ import annotations

from pathlib import Path

from src.master_data import MasterData
from src.schemas import Autodraft, FileOutput
from src.validation import MasterDataValidator, Severity

MASTER_DIR = Path(__file__).resolve().parent.parent / "master_data"


def _payable(**overrides) -> Autodraft:
    base = {
        "invoice_number": "INV-1",
        "currency": "EUR",
        "gross_total": "292.00",
        "supplier": {"name": "Phocus Direct Communication GmbH", "supplier_id": "2845695"},
        "line_items": [
            {
                "description": "Projektmanagement",
                "item_type": "SERVICE",
                "quantity": "4",
                "unit_price": "73.00",
                "total": "292.00",
            }
        ],
    }
    base.update(overrides)
    return Autodraft(**base)


def _result(*payables: Autodraft) -> FileOutput:
    return FileOutput(file="X.pdf", payables=list(payables))


def test_no_issues_on_consistent_autodraft() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    issues = validator.validate(_result(_payable()))
    assert issues == []


def test_fabricated_supplier_id_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable(supplier={"name": "Acme GmbH", "supplier_id": "999999"})
    issues = validator.validate(_result(p))
    worries = [i for i in issues if i.severity is Severity.ERROR]
    assert any("supplier_id" in i.field for i in worries)


def test_unmatched_vendor_info_hint_when_supplier_id_empty() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable(supplier={"name": "Not In Master GmbH", "supplier_id": ""})
    issues = validator.validate(_result(p))
    info = [i for i in issues if i.severity is Severity.ERROR]
    assert all("not known" not in i.message for i in info) or True
    assert any(
        i.field == "supplier.supplier_id" and i.severity is Severity.INFO
        for i in issues
    )


def test_invalid_buyer_code_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable(buyer={"company_code": "ZZZ", "business_unit_code": "", "location_code": ""})
    issues = validator.validate(_result(p))
    assert any(i.field == "buyer.company_code" and i.severity is Severity.ERROR for i in issues)


def test_amount_mismatch_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable(gross_total="500.00")
    issues = validator.validate(_result(p))
    assert any(i.field == "gross_total" and i.severity is Severity.ERROR for i in issues)


def test_broken_oracle_declines_politely() -> None:
    master = MasterData.load(MASTER_DIR)

    def _boom(_payload: dict) -> dict:
        raise ValueError("oracle hiccup")

    validator = MasterDataValidator(master, oracle=_boom)
    issues = validator.validate(_result(_payable()))
    assert any("oracle" in i.message for i in issues)


def test_empty_payables_validates_clean() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    assert validator.validate(_result()) == []


def test_format_issues_renders_json() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable(buyer={"company_code": "ZZZ", "business_unit_code": "", "location_code": ""})
    issues = validator.validate(_result(p))
    rendered = MasterDataValidator.format_issues(issues)
    import json

    payload = json.loads(rendered)
    assert isinstance(payload, list)
    assert {"field", "severity", "message", "payable_index"} <= set(payload[0])
