"""Tests for schema fidelity (:mod:`src.schemas`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from src.schemas import Autodraft, FileOutput, LineItem, Tax

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Product", "GOODS"),
        ("GOODS", "GOODS"),
        ("services", "SERVICE"),
        ("Shipping", "FREIGHT"),
        ("sale tax", "TAX"),
        ("??", "SERVICE"),  # unknown -> safest default
        ("", "SERVICE"),
        (None, "SERVICE"),
    ],
)
def test_item_type_coercion(raw: Any, expected: str) -> None:
    line = LineItem(item_type=raw)
    assert line.item_type == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Credit Note", "CREDIT_MEMO"),
        ("CREDIT MEMO", "CREDIT_MEMO"),
        ("Invoice", "INVOICE"),
        ("RECHNUNG", "INVOICE"),
        ("", "INVOICE"),
    ],
)
def test_invoice_type_coercion(raw: str, expected: str) -> None:
    assert Autodraft(invoice_type=raw).invoice_type == expected


def test_dot_decimal_normalisation() -> None:
    # European: dot thousands, comma decimal
    assert Tax(tax_rate="19 %", tax_amount="1.234,56").tax_amount == "1234.56"
    # English: comma thousands, dot decimal
    assert Tax(tax_rate="0", tax_amount="1,234.56").tax_amount == "1234.56"
    # Bare comma decimals and currency symbols
    assert Tax(tax_amount="€ 123,45").tax_amount == "123.45"
    assert Tax(tax_rate="0").tax_rate == "0"


def test_default_autodraft_is_all_empty_strings() -> None:
    payload = Autodraft().model_dump()
    nested = {"supplier", "buyer", "taxes", "line_items"}
    assert payload["invoice_type"] == "INVOICE"
    for key, value in payload.items():
        if key == "invoice_type" or key in nested:
            continue
        assert value == "", f"{key!r} should default to ''"
    assert payload["line_items"] == []
    assert payload["taxes"] == []


def test_sample_autodraft_round_trips() -> None:
    with (REPO_ROOT / "sample_autodraft.json").open(encoding="utf-8") as fh:
        sample = json.load(fh)
    payable = Autodraft.model_validate(sample)
    out = json.loads(payable.model_dump_json())
    assert out == sample


def test_extra_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        Autodraft(invoice_number="1", hallucinated_field="x")


def test_file_output_serialises_to_contract() -> None:
    result = FileOutput(
        file="X.pdf",
        payables=[Autodraft(invoice_number="42", currency="EUR")],
        declined=[],
    )
    blob = json.loads(result.to_json())
    assert set(blob) == {"file", "payables", "declined"}
    assert blob["file"] == "X.pdf"
    assert blob["payables"][0]["invoice_number"] == "42"
