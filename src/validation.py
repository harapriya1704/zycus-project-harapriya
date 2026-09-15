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

#: Known section-header / subtotal-summary row labels (lowercase). These rows
#: repeat a category name and carry the section's *aggregated* amount — they are
#: not atomic charges and would double-count against the ERP price qty x rate.
_HEADER_PATTERNS: tuple[str, ...] = (
    "peco electric delivery",
    "peco electric",
    "electric supply",
    "electric delivery",
    "taxes & fees",
    "taxes and fees",
    "total new charges",
    "total current charges",
    "total charges",
    "subtotal",
)


def _to_float(value: object) -> float | None:
    """Parse a dot-decimal string to float; ``None`` when unparseable."""
    try:
        return float(str(value or "").strip())
    except ValueError:
        return None


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
        issues += self._check_tax_duplication(payable, index)
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
                        f"({abs(booked_gross - printed):.2f} difference). NOTE: the ERP does "
                        "NOT read the printed subtotal / total_tax_amount / gross_total fields. "
                        "It prices every line_items entry as quantity x unit_price, adds each "
                        "TAX line's tax_amount, then adds header taxes[] — so a tax booked "
                        "both as a TAX line and in header taxes[] is added twice. Align "
                        "line_items + taxes[] so this recompute equals the printed gross; "
                        "keep every tax in EXACTLY ONE place."
                    ),
                    payable_index=idx,
                )
            ]
        return []

    # ------------------------------------------------------------------
    def _check_lines(self, p: Autodraft, idx: int) -> list[ValidationIssue]:
        """Checks for the corrector: line completeness, arithmetic and PO presence."""
        issues: list[ValidationIssue] = []

        for li_no, line in enumerate(p.line_items):
            self._check_line_completeness(line, li_no, idx, issues)
            self._check_header_rows(line, li_no, idx, issues)
            # Line extension should equal qty * unit_price (when both are present).
            qty_f = _to_float(line.quantity)
            price_f = _to_float(line.unit_price)
            if qty_f is not None and price_f is not None:
                computed = qty_f * price_f
                if str(line.total or "").strip():
                    printed = _to_float(line.total)
                    if printed is not None:
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
    def _check_line_completeness(
        line: Any,
        li_no: int,
        idx: int,
        issues: list[ValidationIssue],
    ) -> None:
        """Flag non-tax lines that carry a total but no quantity/unit_price.

        The ERP prices a line as quantity x unit_price; an empty price books
        the line as ZERO, silently under-stating the gross. Tax lines are the
        exception — they are priced from ``tax_amount`` / ``tax_rate``.
        """
        if line.item_type == "TAX":
            return
        total = _to_float(line.total)
        if total is None or total == 0.0:
            return
        if not (str(line.quantity or "").strip() and str(line.unit_price or "").strip()):
            issues.append(
                ValidationIssue(
                    field=f"line_items[{li_no}].unit_price",
                    severity=Severity.ERROR,
                    message=(
                        f"Line {li_no + 1} ('{line.description or ''}'): total {line.total} is set "
                        "but quantity/unit_price is missing. Line items must have explicit "
                        "quantity and unit_price. If missing, set "
                        f"quantity = '1' and unit_price = '{line.total}' (equal to the line total)."
                    ),
                    payable_index=idx,
                )
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _check_header_rows(
        line: Any,
        li_no: int,
        idx: int,
        issues: list[ValidationIssue],
    ) -> None:
        """Flag category-header / subtotal-summary rows as ERROR.

        Headers such as "PECO ELECTRIC DELIVERY", "ELECTRIC SUPPLY" or
        "TAXES & FEES" repeat a section name and carry the section's aggregated
        amount; they are not atomic charges. Keeping them double-counts the
        ERP's qty x price book. On removal the header totals must be
        re-derived, otherwise the ERP oracle check fails.
        """
        name = str(line.description or "").strip().lower()
        if not name:
            return
        flagged = any(pattern in name for pattern in _HEADER_PATTERNS)
        if not flagged:
            return
        issues.append(
            ValidationIssue(
                field=f"line_items[{li_no}]",
                severity=Severity.ERROR,
                message=(
                    f"Line {li_no + 1} ('{line.description}') is a category header / "
                    "subtotal summary row, not an atomic charge. DELETE it from "
                    "line_items. Then recompute the printable totals from the "
                    "remaining lines: subtotal = sum of non-tax line totals, "
                    "total_tax_amount = sum of item_type TAX line totals, and "
                    "gross_total = subtotal + total_tax_amount (rounded to 2 "
                    "decimals)."
                ),
                payable_index=idx,
            )
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _check_tax_duplication(p: Autodraft, idx: int) -> list[ValidationIssue]:
        """Flag taxes booked twice: as a TAX line AND in header taxes[].

        ``erp.erp_book`` adds ``item_type == "TAX"`` rows' ``tax_amount`` and
        then the header ``taxes[]`` amounts — a tax present in both places is
        booked twice. Keep each tax in exactly one of the two slots.
        """
        if not p.taxes:
            return []

        def _norm(s: object) -> str:
            return str(s or "").strip().lower()

        header_keys = {
            (_norm(t.tax_name), round(float(_to_float(t.tax_amount) or 0.0), 2))
            for t in p.taxes
            if _to_float(t.tax_amount) is not None
        }
        issues: list[ValidationIssue] = []
        for li_no, line in enumerate(p.line_items):
            if line.item_type != "TAX":
                continue
            amount = _to_float(line.tax_amount)
            if amount is None:
                amount = _to_float(line.total)
            if amount is None:
                continue
            key = (_norm(line.description), round(float(amount), 2))
            if key in header_keys:
                issues.append(
                    ValidationIssue(
                        field="taxes[]",
                        severity=Severity.ERROR,
                        message=(
                            f"Tax '{line.description}' ({amount:.2f}) is booked twice: "
                            f"as line_items[{li_no}] (item_type TAX) AND in header taxes[]. "
                            "The ERP adds both, inflating gross. Keep it in EXACTLY ONE "
                            "place — remove it from header taxes[] (or, if you intend it "
                            "as a header tax, remove the TAX line row instead)."
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
