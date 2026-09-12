"""Pydantic models mirroring ``AUTODRAFT_SCHEMA.md`` exactly.

These are the single source of truth for the record shape the ERP consumes.
Every field is a **string** and defaults to ``""`` (or ``[]`` for lists) so an
LLM that can only fill part of a record still produces a valid, schema-shaped
JSON object; the ERP parses empties as zero. The schema also drives
structured-output generation (:mod:`src.formatter`) — the model *is* the schema.

Domain rules enforced here (rather than by the LLM):
- ``item_type`` is coerced to one of GOODS / SERVICE / FREIGHT / TAX.
- ``invoice_type`` is coerced to INVOICE / CREDIT_MEMO.
- unknown/redundant keys are rejected (``extra="forbid"``).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Allowed line-item kinds, per AUTODRAFT_SCHEMA.md.
ITEM_TYPES = ("GOODS", "SERVICE", "FREIGHT", "TAX")
#: Allowed payable kinds, per AUTODRAFT_SCHEMA.md.
INVOICE_TYPES = ("INVOICE", "CREDIT_MEMO")
#: Allowed tax_type values seen in ``tax_master.json``.
TAX_TYPES = ("VAT", "IVA", "SST", "NHIL", "GETFL", "COVID", "CST", "HST", "GST", "MOMS", "NP", "WHT", "USE")

# Sensible aliases the LLM might emit for the enums above.
_ITEM_TYPE_ALIASES = {
    "GOODS": "GOODS", "GOOD": "GOODS", "ITEM": "GOODS", "PRODUCT": "GOODS",
    "SERVICE": "SERVICE", "SERVICES": "SERVICE",
    "FREIGHT": "FREIGHT", "SHIPPING": "FREIGHT", "TRANSPORT": "FREIGHT",
    "TAX": "TAX", "CHARGE": "TAX", "LEVY": "TAX",
    "SALE TAX": "TAX", "SALES TAX": "TAX", "FEE": "TAX", "FEES": "TAX",
}
_INVOICE_TYPE_ALIASES = {
    "INVOICE": "INVOICE", "BILL": "INVOICE", "RECHNUNG": "INVOICE", "FATURA": "INVOICE",
    "CREDIT_MEMO": "CREDIT_MEMO", "CREDIT MEMO": "CREDIT_MEMO", "CREDIT": "CREDIT_MEMO",
    "CREDIT NOTE": "CREDIT_MEMO", "CREDITNOTE": "CREDIT_MEMO", "KREDITNOTE": "CREDIT_MEMO",
}


def _to_dot_decimal(value: object) -> str:
    """Normalise a printed money/rate value to a dot-decimal string.

    Handles both locale styles:
    - European ``1.234,56`` -> ``1234.56`` (dot thousands, comma decimal)
    - English  ``1,234.56`` -> ``1234.56`` (comma thousands, dot decimal)
    - Bare commas with 1-2 decimals (``73,00``) are treated as decimal marks.

    Currency symbols, ``%`` and spaces are stripped.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = s.replace("%", "").replace(" ", "").lstrip("€$£¥₹")
    has_dot = "." in s
    has_comma = "," in s
    if has_dot and has_comma:
        # The *last* separator is the decimal mark.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        # A comma followed by 1-2 trailing digits is a decimal mark;
        # otherwise it is a thousands separator ("1,234") and is dropped.
        if re.search(r"\d{3}$", s) or not re.search(r"\d{1,2}$", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    return s


def _coerce(value: object, aliases: dict[str, str], default: str) -> str:
    """Coerce ``value`` through an alias table; fall back to ``default``."""
    raw = str(value or "").strip().upper()
    return aliases.get(raw, default)


class Supplier(BaseModel):
    """The issuer of the document."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    supplier_id: str = ""  # master-data code, "" = no match
    address: str = ""
    vat_id: str = ""


class Buyer(BaseModel):
    """The tenant organisation the document is issued to (from master data)."""

    model_config = ConfigDict(extra="forbid")

    company_code: str = ""
    business_unit_code: str = ""
    location_code: str = ""


class Tax(BaseModel):
    """One tax / levy. Usable at the header or on a line (see schema notes)."""

    model_config = ConfigDict(extra="forbid")

    tax_type: str = ""
    tax_name: str = ""
    tax_rate: str = ""  # percentage, no "%", dot-decimal
    tax_amount: str = ""  # may be "" (ERP derives from rate) or negative
    tax_type_code: str = ""  # master-data code, "" = no match

    @field_validator("tax_rate", "tax_amount", mode="before")
    @classmethod
    def _dot_decimal(cls, v: object) -> str:
        """Normalise a tax rate/amount to a dot-decimal string (see ``_to_dot_decimal``)."""
        return _to_dot_decimal(v)


class LineItem(BaseModel):
    """A raw line component: the ERP derives the net from qty x unit price."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    item_type: str = "SERVICE"  # GOODS | SERVICE | FREIGHT | TAX
    uom: str = ""
    quantity: str = ""
    unit_price: str = ""  # NET (tax-exclusive)
    total: str = ""  # line extension as printed
    discount: str = ""
    discount_percentage: str = ""
    tax_rate: str = ""
    tax_amount: str = ""
    taxes: list[Tax] = Field(default_factory=list)

    @field_validator("item_type", mode="before")
    @classmethod
    def _coerce_item_type(cls, v: object) -> str:
        return _coerce(v, _ITEM_TYPE_ALIASES, "SERVICE")

    @field_validator("quantity", "unit_price", "total", "discount", "discount_percentage", mode="before")
    @classmethod
    def _dot_decimal(cls, v: object) -> str:
        if isinstance(v, (int, float)):
            return f"{v}"
        return _to_dot_decimal(v)


class Autodraft(BaseModel):
    """One bookable payable — a direct mirror of the schema's ``payable``."""

    model_config = ConfigDict(extra="forbid")

    invoice_number: str = ""
    invoice_date: str = ""  # ISO YYYY-MM-DD
    due_date: str = ""
    invoice_type: str = "INVOICE"  # INVOICE | CREDIT_MEMO
    currency: str = ""

    supplier: Supplier = Field(default_factory=Supplier)
    buyer: Buyer = Field(default_factory=Buyer)
    payment_term_id: str = ""  # master-data code
    po_number: str = ""  # as printed
    po_id: str = ""  # matched master-data PO code, "" if not in PO master

    # Totals exactly as printed on the document.
    gross_total: str = ""
    subtotal: str = ""
    total_tax_amount: str = ""

    # Header-level charges & discount (raw components; ERP computes the net).
    discount_amount: str = ""
    freight_charges: str = ""
    insurance_charges: str = ""
    extra_charges: str = ""
    excise_duties: str = ""

    # Header-level taxes.
    taxes: list[Tax] = Field(default_factory=list)

    # Line items: raw components.
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("invoice_type", mode="before")
    @classmethod
    def _coerce_invoice_type(cls, v: object) -> str:
        return _coerce(v, _INVOICE_TYPE_ALIASES, "INVOICE")

    @field_validator("gross_total", "subtotal", "total_tax_amount", mode="before")
    @classmethod
    def _dot_decimal_header(cls, v: object) -> str:
        return _to_dot_decimal(v)


class Declined(BaseModel):
    """A document (or page cluster) deemed not a bookable payable."""

    model_config = ConfigDict(extra="forbid")

    doc_type: str = ""
    reason: str = ""


class FileOutput(BaseModel):
    """The per-file output contract: one object per input PDF."""

    model_config = ConfigDict(extra="forbid")

    file: str = ""
    payables: list[Autodraft] = Field(default_factory=list)
    declined: list[Declined] = Field(default_factory=list)

    def to_json(self) -> str:
        """Serialise to the exact JSON shape the grader/ERP consumes."""
        return self.model_dump_json(indent=2)


# Re-export the narrow type consumers import.
__all__ = [
    "Autodraft",
    "Buyer",
    "Declined",
    "FileOutput",
    "INVOICE_TYPES",
    "ITEM_TYPES",
    "LineItem",
    "Supplier",
    "TAX_TYPES",
    "Tax",
]
