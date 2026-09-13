"""Tests for the multi-agent validation loop (:mod:`src.agents`).

The loop is exercised with fake proposer/corrector runnables so no API key is
needed — the point is the *orchestration*: propose -> validate -> correct with
a hard ``max_retries`` cap.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import RunnableLambda
from src.agents import ValidationLoop
from src.master_data import MasterData
from src.schemas import Autodraft, FileOutput

MASTER_DIR = Path(__file__).resolve().parent.parent / "master_data"

_DOC_TEXT = "Invoice 265345, Phocus Direct Communication GmbH, DE209177122, gross 438.00 EUR"


def _clean_autodraft() -> Autodraft:
    return Autodraft(
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


def _sample_chain() -> RunnableLambda:
    return RunnableLambda(lambda inputs: FileOutput(file="INV-99.pdf", payables=[_clean_autodraft()]))


def test_loop_converges_first_pass_no_corrections() -> None:
    master = MasterData.load(MASTER_DIR)
    loop = ValidationLoop(master=master, max_retries=3, structuring_chain=_sample_chain())
    result = loop.run(_DOC_TEXT, "INV-99.pdf")
    assert result.converged is True
    assert result.attempts == 0
    assert len(result.file_output.payables) == 1
    assert result.file_output.payables[0].supplier.supplier_id == "2845695"


def test_loop_bounded_corrections_persist_issues() -> None:
    master = MasterData.load(MASTER_DIR)

    def _bad_proposer(_inputs: dict) -> FileOutput:
        return FileOutput(
            file="INV-99.pdf",
            payables=[Autodraft(invoice_number="1", currency="EUR", gross_total="999.00",
                                buyer={"company_code": "ZZZ", "business_unit_code": "", "location_code": ""})],
        )

    def _never_fixes(_inputs: dict) -> Autodraft:
        # Returns the same broken payable regardless of the issues it saw.
        return Autodraft(invoice_number="1", currency="EUR", gross_total="999.00")

    loop = ValidationLoop(
        master=master,
        max_retries=3,
        structuring_chain=RunnableLambda(_bad_proposer),
        corrector_chain=RunnableLambda(_never_fixes),
    )
    result = loop.run(_DOC_TEXT, "INV-99.pdf")
    assert result.attempts == 3  # hard cap, no infinite generation
    assert result.converged is False
    assert loop.stats["corrections"] == 3
    assert loop.stats["not_converged"] == 1


def test_loop_converges_after_correction() -> None:
    master = MasterData.load(MASTER_DIR)

    def _bad_proposer(_inputs: dict) -> FileOutput:
        return FileOutput(
            file="INV-99.pdf",
            payables=[Autodraft(invoice_number="1", currency="EUR", gross_total="999.00")],
        )

    def _fixes(_inputs: dict) -> Autodraft:
        return _clean_autodraft()

    loop = ValidationLoop(
        master=master,
        max_retries=3,
        structuring_chain=RunnableLambda(_bad_proposer),
        corrector_chain=RunnableLambda(_fixes),
    )
    result = loop.run(_DOC_TEXT, "INV-99.pdf")
    assert result.attempts == 1
    assert result.converged is True
    assert result.file_output.payables[0].supplier.supplier_id == "2845695"


def test_max_retries_cap_respected_with_zero_retries() -> None:
    master = MasterData.load(MASTER_DIR)

    def _bad_proposer(_inputs: dict) -> FileOutput:
        return FileOutput(file="INV-99.pdf",
                          payables=[Autodraft(invoice_number="1", currency="EUR", gross_total="999.00")])

    def _never_fixes(_inputs: dict) -> Autodraft:
        return Autodraft(invoice_number="1", currency="EUR", gross_total="999.00")

    loop = ValidationLoop(
        master=master,
        max_retries=1,
        structuring_chain=RunnableLambda(_bad_proposer),
        corrector_chain=RunnableLambda(_never_fixes),
    )
    result = loop.run(_DOC_TEXT, "INV-99.pdf")
    assert result.attempts in (0, 1)
    assert result.attempts <= 1


def test_declined_documents_never_enter_corrector() -> None:
    """A declined (no payable) document must produce no correction calls."""
    master = MasterData.load(MASTER_DIR)
    calls: list[dict] = []

    def _declining_proposer(_inputs: dict) -> FileOutput:
        from src.schemas import Declined

        return FileOutput(
            file="DU-01.pdf",
            declined=[Declined(doc_type="STATEMENT", reason="Not a payable.")],
        )

    def _corrector(inputs: dict):
        calls.append(inputs)
        return _clean_autodraft()

    loop = ValidationLoop(
        master=master,
        max_retries=3,
        structuring_chain=RunnableLambda(_declining_proposer),
        corrector_chain=RunnableLambda(_corrector),
    )
    result = loop.run(_DOC_TEXT, "DU-01.pdf")
    assert result.converged is True
    assert result.file_output.payables == []
    assert len(result.file_output.declined) == 1
    assert calls == []
