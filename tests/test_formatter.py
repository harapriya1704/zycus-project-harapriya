"""Tests for the structuring/resolving stage (:mod:`src.formatter`)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableLambda
from src.formatter import build_autodraft_chain, format_autodraft, resolve_codes
from src.master_data import MasterData
from src.schemas import Autodraft, Declined, FileOutput

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fake_chain(payload: FileOutput):
    return RunnableLambda(lambda _: payload)


def test_empty_text_is_declined() -> None:
    result = format_autodraft(
        document_text="   ",
        filename="X.pdf",
        chain=_fake_chain(FileOutput()),
        master=MasterData(),
    )
    assert result.file == "X.pdf"
    assert result.payables == []
    assert result.declined == [Declined(doc_type="UNREADABLE", reason=result.declined[0].reason)]


def test_chain_sets_file_and_codes() -> None:
    llm_out = FileOutput(
        file="",
        payables=[
            Autodraft(
                invoice_number="265345",
                currency="EUR",
                supplier={"name": "Phocus Direct Communication GmbH", "vat_id": "DE209177122"},
            )
        ],
    )
    result = format_autodraft(
        document_text="some invoice text",
        filename="A.pdf",
        chain=_fake_chain(llm_out),
        master=MasterData.load(REPO_ROOT / "master_data"),
    )
    assert result.file == "A.pdf"
    assert result.payables[0].supplier.supplier_id == "2845695"


def test_unknown_supplier_stays_blank() -> None:
    llm_out = FileOutput(
        payables=[Autodraft(invoice_number="1", supplier={"name": "Acme Neverland"})]
    )
    result = resolve_codes(llm_out, MasterData.load(REPO_ROOT / "master_data"), document_text="Acme Neverland")
    assert result.payables[0].supplier.supplier_id == ""


def test_tax_type_code_resolved() -> None:
    llm_out = FileOutput(
        payables=[
            Autodraft(
                invoice_number="1",
                currency="EUR",
                supplier={"vat_id": "DE209177122"},
                taxes=[{"tax_type": "VAT", "tax_name": "VAT Reverse Charge", "tax_rate": "0"}],
            )
        ]
    )
    result = resolve_codes(
        llm_out,
        MasterData.load(REPO_ROOT / "master_data"),
        document_text="VAT Reverse Charge",
    )
    assert result.payables[0].taxes[0].tax_type_code == "DE_000_RC"


def test_build_chain_requires_api_key(monkeypatch) -> None:

    import pytest

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        build_autodraft_chain(model="gpt-4o-mini", settings=None)
