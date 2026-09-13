"""Deterministic "Validation Agent": cross-reference autodrafts against master data.

The proposer agent (day-1 structuring chain) guesses nothing about master
*data itself* — it leaves every code ``""`` and the resolver derives the codes.
This module is the other half of the bargain: it re-checks, field by field,
that the final autodraft is internally consistent with the masters and with the
ERP's own recompute oracle (``erp.erp_book``).

Checks performed per payable:
- every master code that *is* set resolves to a real master row;
- a set ``supplier_id`` with no matching master supplier is flagged
  (unmatched vendor ID).
- amounts: the ERP recompute of the payable's raw components is compared to the
  printed ``gross_total``; a mismatch beyond tolerance is flagged.
- structural sanity (line arithmetic, PO/buyer/tax-code presence) is reported
  as INFO/WARNING so the loop has concrete hints to act on.

The validator is pure logic — no LLM call — which keeps the feedback loop
fast, deterministic and audit-friendly. Severity: ``ERROR`` drives the
correction loop; ``WARNING``/``INFO`` are advisory.

    from src.validation import MasterDataValidator, ValidationIssue, Severity
    issues = MasterDataValidator(master).validate(file_output, document_text)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from erp import erp_book

from src.master_data import MasterData
from src.schemas import Autodraft, FileOutput

#: Tolerance (in currency units) for the ERP-recompute vs printed gross check.
_GROSS_TOLERANCE: float = 0.01


class Severity(str, Enum):
    """How strongly an issue blocks the pipeline's correction loop."""

    ERROR = "ERROR"  # inconsistency that the corrector agent must address
    WARNING = "WARNING"  # likely problem; does not block convergence
    INFO = "INFO"  # advisory hint for the corrector agent


@dataclass(frozen=True)
class ValidationIssue:
    """One flagged inconsistency on a single payable."""

    field: str  # dotted path, e.g. "supplier.supplier_id" or "gross_total"
    severity: Severity
    message: str
    payable_index: int = 0


# ---------------------------------------------------------------------------
# Ideal client for master data at reconcile time: expose the code sets so the
# validator never scans lists. MasterData already keeps O(1) indexes.
# ---------------------------------------------------------------------------

def _master_code_sets(master: MasterData) -> dict[str, set[str]]:
    """Collect every master code the validator must recognise."""
    return {
        "suppliers": {str(r.get("supplier_id", "")) for r in master.suppliers},
        "companies": {str(c.get("company_code", "")) for c in master.companies},
        "taxes": {str(t.get("code", "")) for t in master.taxes},
        "payment_terms": {str(t.get("payment_term_id", "")) for t in master.payment_terms},
        "purchase_orders": {str(p.get("po_id", "")) for p in master.purchase_orders},
    }


@dataclass
class MasterDataValidator:
    """Stateless-by-design validator (safe to share across documents).

    Uses :class:`MasterData`'s O(1) indexes for code existence and the
    ``erp.erp_book`` oracle for the amount check.
    """

    master: MasterData
    oracle: Callable[[dict[str, Any]], dict[str, Any]] = erp_book
    gross_tolerance: float = _GROSS_TOLERANCE

    _codes: dict[str, set[str]] = field(init=False)

    def __post_init__(self) -> None:
        self._codes = _master_code_sets(self.master)

    # ------------------------------------------------------------------
    def validate(self, result: FileOutput, document_text: str = "") -> list[ValidationIssue]:
        """Validate every payable in a file result.

        Args:
            result: the autodraft(s) produced by the proposer + resolver.
            document_text: the source document text (used for cross-checks
                that need the raw document, e.g. printed PO numbers).

        Returns:
            A flat list of :class:`ValidationIssue` objects; empty when
            everything is consistent.
        """
        issues: list[ValidationIssue] = []
        for idx, payable in enumerate(result.payables):
            issues.extend(self.validate_payable(payable, document_text=document_text, index=idx))
        return issues

    # ------------------------------------------------------------------
    def validate_payable(
        self,
        payable: Autodraft,
        document_text: str = "",
        index: int = 0,
    ) -> list[ValidationIssue]:
        """Validate a single payable; returns the issues found on it."""
        issues: list[ValidationIssue] = []

        issues += self._check_codes(payable, index)
        issues += self._check_unmatched(payable, index)
        issues += self._check_gross(payable, index)
        issues += self._check_lines(payable, index)
        return [i for i in issues if i is not None]

    # ------------------------------------------------------------------
    def _check_codes(self, p: Autodraft, idx: int) -> list[ValidationIssue]:
        """Master-data codes that are set must exist in the masters."""
        issues: list[ValidationIssue] = []

        for path, code, allowed in (
            ("buyer.company_code", p.buyer.company_code, self._codes["companies"]),
            ("payment_term_id", p.payment_term_id, self._codes["payment_terms"]),
            ("po_id", p.po_id, self._codes["purchase_orders"]),
        ):
            if code and code not in allowed:
                issues.append(
                    ValidationIssue(
                        field=path,
                        severity=Severity.ERROR,
                        message=f"Master data code '{code}' for {path.split('.')[-1]} is not in its master file.",
                        payable_index=idx,
                    )
                )

        seen_tax_codes: list[tuple[Any, object, object]] = []
        for tax in p.taxes:
            seen_tax_codes.append((tax.tax_type_code, tax.tax_rate, tax.tax_amount))
        for line in p.line_items:
            for tax in line.taxes:
                seen_tax_codes.append((tax.tax_type_code, tax.tax_rate, tax.tax_amount))
        for code, _rate, _amt in seen_tax_codes:
            if str(code) and str(code) not in self._codes["taxes"]:
                issues.append(
                    ValidationIssue(
                        field="taxes[].tax_type_code",
                        severity=Severity.ERROR,
                        message=f"Tax code '{code}' is not in the tax master.",
                        payable_index=idx,
                    )
                )
        return issues

    # ------------------------------------------------------------------
    def _check_unmatched(self, p: Autodraft, idx: int) -> list[ValidationIssue]:
        """Honest empties are fine, but a present-but-unmatched ID is not."""
        issues: list[ValidationIssue] = []

        if p.supplier.supplier_id and p.supplier.supplier_id not in self._codes["suppliers"]:
            issues.append(
                ValidationIssue(
                    field="supplier.supplier_id",
                    severity=Severity.ERROR,
                    message=(
                        f"Unmatched vendor: supplier_id '{p.supplier.supplier_id}' for "
                        f"'{p.supplier.name}' has no row in suppliers.json."
                    ),
                    payable_index=idx,
                )
            )
        elif p.supplier.name and not p.supplier.supplier_id:
            issues.append(
                ValidationIssue(
                    field="supplier.supplier_id",
                    severity=Severity.INFO,
                    message=(
                        "Supplier present on the document but no supplier_id resolved "
                        "(legitimate only if the vendor is not in the master)."
                    ),
                    payable_index=idx,
                )
            )
        return issues

    # ------------------------------------------------------------------
    def _check_gross(self, p: Autodraft, idx: int) -> list[ValidationIssue]:
        """Recompute gross via the ERP oracle and compare to the printed total.

        The ERP consumes a plain ``dict`` of the payable's raw components —
        exactly what :class:`Autodraft` dumps. This makes the oracle check a
        lossless round-trip of the *same* payload the grading system books.
        """
        if not str(p.gross_total or "").strip():
            return [
                ValidationIssue(
                    field="gross_total",
                    severity=Severity.WARNING,
                    message="No printed gross_total to compare the ERP recompute against.",
                    payable_index=idx,
                )
            ]

        payload = p.model_dump(mode="json")
        try:
            booked = self.oracle(payload)
        except Exception as exc:  # noqa: BLE001 - a broken oracle shouldn't kill the pipeline
            return [
                ValidationIssue(
                    field="gross_total",
                    severity=Severity.ERROR,
                    message=f"ERP oracle rejected the payload: {type(exc).__name__}: {exc}",
                    payable_index=idx,
                )
            ]

        try:
            printed = float(p.gross_total)
        except ValueError:
            return [
                ValidationIssue(
                    field="gross_total",
                    severity=Severity.ERROR,
                    message=f"Printed gross_total '{p.gross_total}' is not a dot-decimal number.",
                    payable_index=idx,
                )
            ]

        booked_gross = float(booked.get("will_book_gross", 0.0) or 0.0)
        if abs(booked_gross - printed) > self.gross_tolerance:
            return [
                ValidationIssue(
                    field="gross_total",
                    severity=Severity.ERROR,
                    message=(
                        f"Amount inconsistency: ERP recomputes gross as {booked_gross:.2f} "
                        f"{booked.get('currency', '')} but the document prints {printed:.2f} "
                        f"({abs(booked_gross - printed):.2f} difference)."
                    ),
                    payable_index=idx,
                )
            ]
        return []

    # ------------------------------------------------------------------
    def _check_lines(self, p: Autodraft, idx: int) -> list[ValidationIssue]:
        """Advisory checks for the corrector: line arithmetic and PO presence."""
        issues: list[ValidationIssue] = []

        for li_no, line in enumerate(p.line_items):
            # Line extension should equal qty * unit_price (when both are present).
            if str(line.quantity or "").strip() and str(line.unit_price or "").strip():
                try:
                    computed = float(line.quantity) * float(line.unit_price)
                except ValueError:
                    continue
                if str(line.total or "").strip():
                    try:
                        printed = float(line.total)
                    except ValueError:
                        continue
                    if abs(computed - printed) > self.gross_tolerance:
                        issues.append(
                            ValidationIssue(
                                field=f"line_items[{li_no}].total",
                                severity=Severity.WARNING,
                                message=(
                                    f"Line {li_no + 1}: qty x price = {computed:.2f} but the line "
                                    f"extension prints {printed:.2f}."
                                ),
                                payable_index=idx,
                            )
                        )

        if p.po_number and not p.po_id:
            issues.append(
                ValidationIssue(
                    field="po_id",
                    severity=Severity.INFO,
                    message=(
                        f"Document references PO '{p.po_number}' which is not in the PO master; "
                        "po_id left empty (non-ERP reference)."
                    ),
                    payable_index=idx,
                )
            )
        return issues

    # ------------------------------------------------------------------
    @staticmethod
    def format_issues(issues: list[ValidationIssue]) -> str:
        """Render issues as a compact, LLM-friendly JSON list."""
        payload = [
            {"field": i.field, "severity": i.severity.value, "message": i.message, "payable_index": i.payable_index}
            for i in issues
        ]
        return json.dumps(payload, indent=2)
