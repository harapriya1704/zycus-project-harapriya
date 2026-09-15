"""Tests for the structuring/resolving stage (:mod:`src.formatter`)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableLambda
from src.formatter import (
    build_autodraft_chain,
    format_autodraft,
    resolve_codes,
)
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
    result = resolve_codes(
        llm_out, MasterData.load(REPO_ROOT / "master_data"), document_text="Acme Neverland"
    )
    assert result.payables[0].supplier.supplier_id == ""


def test_oversized_transcript_truncation_keeps_head_and_tail() -> None:
    from src.config import Settings

    captured: dict[str, str] = {}

    def _probe(payload):
        captured["document_text"] = payload["document_text"]
        return FileOutput()

    head = "HEAD-" * 50  # 250 chars
    body = "M" * 600  # 600 chars
    tail = "TAIL-" * 60  # 300 chars
    long_text = head + body + tail  # 1150 chars > 400-char cap

    result = format_autodraft(
        document_text=long_text,
        filename="BIG.pdf",
        chain=RunnableLambda(_probe),
        master=MasterData(),
        settings=Settings(max_structuring_chars=400),
    )
    # The LLM only ever sees a bounded window, but that window keeps BOTH the
    # document head (header / supplier) and the tail (line items / totals).
    assert len(captured["document_text"]) <= 400
    assert captured["document_text"].startswith("HEAD-")
    assert captured["document_text"].endswith("TAIL-")
    assert "MIDDLE OMITTED" in captured["document_text"]
    # ...while the deterministic resolver still receives the full transcript.
    assert result.file == "BIG.pdf"


def test_save_truncate_returns_text_when_fits() -> None:
    from src.formatter import _save_truncate

    text = "short invoice text"
    assert _save_truncate(text, 100) == text


def test_save_truncate_split_ratios() -> None:
    from src.formatter import _save_truncate

    head = "A" * 200
    body = "B" * 500
    tail = "C" * 200
    truncated = _save_truncate(head + body + tail, 300)
    assert len(truncated) <= 300
    assert truncated.startswith("A" * 10)
    assert truncated.endswith("C" * 10)
    assert "B" not in truncated  # the middle is what gets omitted


def test_looks_like_billable_customs_detects_amounts() -> None:
    from src.formatter import _looks_like_billable_customs

    assert _looks_like_billable_customs(
        "CUSTOMS INVOICE - IMPORT DUTY EUR 450.00, BROKERAGE 80.00"
    )
    assert _looks_like_billable_customs("DUTY INVOICE total 1,234.56 USD")


def test_looks_like_billable_customs_ignores_bare_declarations() -> None:
    from src.formatter import _looks_like_billable_customs

    assert not _looks_like_billable_customs(
        "CUSTOMS DECLARATION - no duty due, no charges, no amounts listed"
    )
    assert not _looks_like_billable_customs("plain invoice without customs words")


def test_customs_billable_decline_is_restructured(monkeypatch) -> None:
    from src.config import Settings
    from src.formatter import (
        _CUSTOMS_RESTRUCTURE_NOTE,
        _looks_like_billable_customs,
        format_autodraft,
    )
    from src.schemas import Declined

    def _first_pass(_inputs) -> FileOutput:
        return FileOutput(
            file="",
            declined=[Declined(doc_type="CUSTOMS_INVOICE", reason="customs not payable")],
        )

    captured: dict[str, str] = {}

    def _retry_chain_factory(**kwargs) -> RunnableLambda:
        captured["system_prompt"] = kwargs.get("system_prompt", "")
        return RunnableLambda(lambda _i: FileOutput(
            file="",
            payables=[
                Autodraft(
                    invoice_number="IMP-1", currency="EUR", gross_total="1234.56",
                    excise_duties="450.00",
                )
            ],
        ))

    monkeypatch.setattr("src.formatter.build_autodraft_chain", _retry_chain_factory)

    doc_text = "CUSTOMS INVOICE - IMPORT DUTY EUR 450.00, BROKERAGE 80.00, TOTAL EUR 1234.56"
    assert _looks_like_billable_customs(doc_text)
    result = format_autodraft(
        document_text=doc_text,
        filename="IMP-1.pdf",
        chain=RunnableLambda(_first_pass),
        master=MasterData(),
        settings=Settings(fast_mode=False),
    )
    assert len(result.payables) == 1
    assert result.payables[0].invoice_number == "IMP-1"
    assert result.payables[0].excise_duties == "450.00"
    assert result.file == "IMP-1.pdf"
    # The corrective instruction was appended to the stock system prompt.
    assert _CUSTOMS_RESTRUCTURE_NOTE.strip() in captured["system_prompt"]


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


def test_build_chain_requires_api_key(monkeypatch, tmp_path) -> None:

    import pytest

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)

    from src.config import Settings

    # Build a keyless settings object and force the singleton to return it, so
    # the credentials present in .env cannot leak into this hermetic test.
    def _keyless() -> Settings:
        return Settings(groq_api_key="", hf_token="", llm_api_key="", llm_base_url="")

    monkeypatch.setattr("src.formatter.get_settings", _keyless)
    with pytest.raises(RuntimeError, match="credentials"):
        build_autodraft_chain(model="gpt-4o-mini", settings=None)
