"""Integration test: the real LangChain structuring chain wired end-to-end.

Drives the chain built by :func:`src.formatter.build_autodraft_chain`
(prompt + ``with_structured_output``) end-to-end, with the network boundary
replaced: ``ChatOpenAI.with_structured_output`` is stubbed so the schema is
still the real :class:`src.schemas.FileOutput` and responses are still
Pydantic-validated. This catches real composition regressions (e.g. the
system prompt's literal ``{}`` breaking prompt templating) without an API key.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import src.formatter as formatter
from langchain_core.runnables import RunnableLambda
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


@pytest.fixture
def stub_structured_output(monkeypatch) -> None:
    """Replace the OpenAI structured-output call with a schema-validated stub."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-really-used")

    def _fake_with_structured(self, schema, *, method=None, **kwargs):
        def _run(_inputs: dict) -> FileOutput:
            return schema.model_validate_json(_SAMPLE_RESPONSE)

        return RunnableLambda(_run, name="stubbed_structured_output")

    monkeypatch.setattr(formatter.ChatOpenAI, "with_structured_output", _fake_with_structured)


def test_chain_composes_and_validates(stub_structured_output) -> None:
    chain = formatter.build_autodraft_chain(model="gpt-4o-mini", temperature=0.0)
    result: FileOutput = chain.invoke({"document_text": "Some invoice text"})
    assert len(result.payables) == 1
    payable = result.payables[0]
    assert payable.invoice_number == "265345"
    assert payable.invoice_type == "INVOICE"
    assert payable.taxes[0].tax_type_code == ""  # codes must NOT be LLM-guessed
    assert payable.line_items[0].item_type == "SERVICE"


def test_format_autodraft_end_to_end_resolves_codes(stub_structured_output) -> None:
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
