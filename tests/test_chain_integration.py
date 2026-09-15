"""Integration test: the real LangChain structuring chain wired end-to-end.

Drives the chain built by :func:`src.formatter.build_autodraft_chain`
(prompt -> chat model -> pydantic parser) end-to-end, with the network
boundary replaced: ``ChatOpenAI.invoke`` is stubbed so the schema is still the
real :class:`src.schemas.FileOutput` and responses are still Pydantic-validated
by the parser. This catches real composition regressions (e.g. the system
prompt's literal ``{}`` breaking prompt templating) without an API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import src.formatter as formatter
from langchain_core.messages import AIMessage
from src.master_data import MasterData
from src.schemas import FileOutput

_SAMPLE_RESPONSE = json.dumps(
    {
        "payables": [
            {
                "invoice_number": "265345",
                "invoice_date": "2026-02-02",
                "due_date": "2026-02-12",
                "invoice_type": "INVOICE",
                "currency": "EUR",
                "supplier": {
                    "name": "Phocus Direct Communication GmbH",
                    "supplier_id": "",
                    "address": "Lina-Ammon-Strasse 19b, 90471 Nurnberg, DE",
                    "vat_id": "DE209177122",
                },
                "buyer": {"company_code": "", "business_unit_code": "", "location_code": ""},
                "payment_term_id": "",
                "po_number": "",
                "po_id": "",
                "gross_total": "438.00",
                "subtotal": "438.00",
                "total_tax_amount": "0.00",
                "discount_amount": "",
                "freight_charges": "",
                "insurance_charges": "",
                "extra_charges": "",
                "excise_duties": "",
                "taxes": [
                    {
                        "tax_type": "VAT",
                        "tax_name": "VAT Reverse Charge",
                        "tax_rate": "0",
                        "tax_amount": "0.00",
                        "tax_type_code": "",
                    }
                ],
                "line_items": [
                    {
                        "description": "Projektmanagement",
                        "item_type": "SERVICE",
                        "uom": "Hr",
                        "quantity": "4",
                        "unit_price": "73.00",
                        "total": "292.00",
                        "discount": "",
                        "discount_percentage": "",
                        "tax_rate": "",
                        "tax_amount": "",
                        "taxes": [],
                    }
                ],
            }
        ],
        "declined": [],
    }
)


def _stub_invoke(self, messages):
    """Return the canned sample JSON for ANY message list."""
    return AIMessage(content=_SAMPLE_RESPONSE)


@pytest.fixture
def stub_chat(monkeypatch) -> None:
    """Replace the network-bound chat call with a canned JSON reply."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-really-used")
    monkeypatch.setattr(formatter.ChatOpenAI, "invoke", _stub_invoke)


def test_chain_composes_and_validates(stub_chat) -> None:
    chain = formatter.build_autodraft_chain(model="gpt-4o-mini", temperature=0.0)
    result: FileOutput = chain.invoke({"document_text": "Some invoice text"})
    assert len(result.payables) == 1
    payable = result.payables[0]
    assert payable.invoice_number == "265345"
    assert payable.invoice_type == "INVOICE"
    assert payable.taxes[0].tax_type_code == ""  # codes must NOT be LLM-guessed
    assert payable.line_items[0].item_type == "SERVICE"


def test_format_autodraft_end_to_end_resolves_codes(stub_chat) -> None:
    master = MasterData.load(Path(__file__).resolve().parent.parent / "master_data")
    result = formatter.format_autodraft(
        _SAMPLE_RESPONSE,
        filename="INV-99.pdf",
        chain=None,
        master=master,
    )
    assert result.file == "INV-99.pdf"
    assert result.payables[0].supplier.supplier_id == "2845695"
    assert result.payables[0].taxes[0].tax_type_code == "DE_000_RC"


def test_parser_tolerates_missing_declined(monkeypatch) -> None:
    """Omitted defaulted fields (common on smaller providers) must fall back."""
    combo = formatter._finalize(FileOutput)
    result = combo.invoke('{"file":"X.pdf","payables":[]}')
    assert result.file == "X.pdf"
    assert result.declined == []  # default restored by the parser


def test_parser_drops_undeclared_fields() -> None:
    """Invented fields (e.g. buyer.name on some models) must be dropped, not fatal."""
    messy = (
        '{"file":"X.pdf","payables":[{"invoice_number":"1","currency":"EUR",'
        '"supplier":{"name":"Acme","supplier_id":"","fax":"+1-555","vat_id":""},'
        '"buyer":{"company_code":"","business_unit_code":"","location_code":"","name":"Buyer Ltd","address":"1 St"}}],'
        '"declined":[]}'
    )
    combo = formatter._finalize(FileOutput)
    result = combo.invoke(messy)
    p = result.payables[0]
    assert p.supplier.name == "Acme"
    assert not hasattr(p.supplier, "fax")
    assert not hasattr(p.buyer, "name")
    assert not hasattr(p.buyer, "address")
