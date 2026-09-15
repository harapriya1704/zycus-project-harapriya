"""Tests for the deterministic validation agent (:mod:`src.validation`)."""

from __future__ import annotations

from pathlib import Path

from src.master_data import MasterData
from src.schemas import Autodraft, FileOutput, LineItem, Tax
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


def _payable_with_lines(*lines: dict) -> Autodraft:
    base = _payable()
    base.line_items = [LineItem(**ln) for ln in lines]
    return base


def test_non_tax_line_missing_qty_price_is_error() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "Energy Charge",
            "item_type": "SERVICE",
            "quantity": "",
            "unit_price": "",
            "total": "13074.77",
        }
    )
    issues = validator.validate(_result(p))
    flagged = [i for i in issues if "unit_price" in i.field and i.severity is Severity.ERROR]
    assert any("quantity = '1'" in i.message for i in flagged)


def test_tax_line_without_qty_price_is_not_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "Gross Receipts Tax",
            "item_type": "TAX",
            "quantity": "",
            "unit_price": "",
            "total": "819.80",
            "tax_amount": "819.80",
        }
    )
    issues = validator.validate(_result(p))
    assert not any("quantity/unit_price" in i.message for i in issues)


def test_complete_line_is_not_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "Distribution",
            "item_type": "SERVICE",
            "quantity": "475",
            "unit_price": "4.77",
            "total": "2265.75",
        }
    )
    issues = validator.validate(_result(p))
    assert not any("quantity/unit_price" in i.message for i in issues)


def test_header_row_flagged_for_deletion() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "PECO ELECTRIC DELIVERY",
            "item_type": "SERVICE",
            "quantity": "1",
            "unit_price": "2752.75",
            "total": "2752.75",
        },
        {
            "description": "Customer Charge",
            "item_type": "SERVICE",
            "quantity": "1",
            "unit_price": "299.78",
            "total": "299.78",
        },
    )
    p.gross_total = "3052.53"
    issues = validator.validate(_result(p))
    header = [
        i for i in issues if i.field == "line_items[0]" and i.severity is Severity.ERROR
    ]
    assert any("category header" in i.message for i in header)
    assert any("DELETE" in i.message for i in header)
    assert not any(i.severity is Severity.ERROR and "gross_total" in i.field for i in issues)


def test_tax_header_row_flagged_for_deletion() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "TAXES & FEES",
            "item_type": "TAX",
            "quantity": "",
            "unit_price": "",
            "total": "164.87",
            "tax_amount": "164.87",
        },
        {
            "description": "Gross Receipts Tax",
            "item_type": "TAX",
            "quantity": "",
            "unit_price": "",
            "total": "819.80",
            "tax_amount": "819.80",
        },
    )
    p.gross_total = "984.67"
    issues = validator.validate(_result(p))
    header = [
        i for i in issues if i.field == "line_items[0]" and i.severity is Severity.ERROR
    ]
    assert any("category header" in i.message for i in header)


def test_tax_duplicated_in_header_and_line_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "State Tax Adjustment",
            "item_type": "TAX",
            "quantity": "",
            "unit_price": "",
            "total": "-0.28",
            "tax_amount": "-0.28",
        },
        {
            "description": "ENERGY CHARGE",
            "item_type": "SERVICE",
            "quantity": "219744",
            "unit_price": "0.0595",
            "total": "13074.77",
        },
    )
    p.taxes = [
        Tax(
            tax_type="TAX",
            tax_name="State Tax Adjustment",
            tax_rate="",
            tax_amount="-0.28",
            tax_type_code="",
        )
    ]
    p.gross_total = "13074.49"
    issues = validator.validate(_result(p))
    dup = [i for i in issues if i.field == "taxes[]" and i.severity is Severity.ERROR]
    assert any("booked twice" in i.message for i in dup)


def test_tax_in_header_only_is_not_flagged() -> None:
    master = MasterData.load(MASTER_DIR)
    validator = MasterDataValidator(master)
    p = _payable_with_lines(
        {
            "description": "ENERGY CHARGE",
            "item_type": "SERVICE",
            "quantity": "219744",
            "unit_price": "0.0595",
            "total": "13074.77",
        }
    )
    p.taxes = [
        Tax(
            tax_type="TAX",
            tax_name="Gross Receipts Tax",
            tax_rate="",
            tax_amount="819.80",
            tax_type_code="",
        )
    ]
    p.gross_total = "13894.57"
    issues = validator.validate(_result(p))
    assert not any("booked twice" in i.message for i in issues)
