"""Tests for master-data resolution (:mod:`src.master_data`).

These run against the real ``master_data/`` files shipped with the repo.
"""

from __future__ import annotations

from pathlib import Path

from src.master_data import MasterData, parse_iso_date
from src.schemas import Supplier

REPO_ROOT = Path(__file__).resolve().parent.parent


def _master() -> MasterData:
    return MasterData.load(REPO_ROOT / "master_data")


def test_supplier_match_by_vat_id() -> None:
    md = _master()
    sid = md.match_supplier(vat_id="DE209177122")
    assert sid == "2845695"


def test_supplier_match_by_name() -> None:
    md = _master()
    assert md.match_supplier(name="Ehast Koiduni OU") == "2807582"
    # fuzzy: drop the legal-entity suffix
    assert md.match_supplier(name="Ehast Koiduni") == "2807582"


def test_supplier_no_match_is_honest_empty() -> None:
    md = _master()
    assert md.match_supplier(name="Nonexistent Ventures GmbH") == ""


def test_tax_match_by_rate_and_country() -> None:
    md = _master()
    assert md.match_tax(tax_type="VAT", rate="19", country="DE") == "DE_190_VAT"
    assert md.match_tax(tax_type="VAT", rate="24", country="EE") == "EST_240_VAT"


def test_tax_match_reverse_charge_zero_rate() -> None:
    md = _master()
    assert (
        md.match_tax(tax_type="VAT", tax_name="VAT Reverse Charge", rate="0", country="DE")
        == "DE_000_RC"
    )


def test_payment_term_by_alias_and_dates() -> None:
    md = _master()
    assert md.match_payment_term(text="Payment terms: NET 10") == "Net_10"
    assert md.match_payment_term(invoice_date="2026-02-02", due_date="2026-02-12") == "Net_10"


def test_po_match() -> None:
    md = _master()
    assert md.match_po("PO-EE-2026-0044") == "PO-EE-2026-0044"
    assert md.match_po("PO-9999-UNKNOWN") == ""


def test_buyer_match_from_document_text() -> None:
    md = _master()
    company, bu, loc = md.match_buyer(text="Invoice to Bolt Holdings OU, Vana-Louna 15")
    assert (company, bu, loc) == ("BOLTGROUP", "EE004", "LOC_EE_001")


def test_buyer_no_match() -> None:
    md = _master()
    assert md.match_buyer(text="tiny unknown supplier address ltd") == ("", "", "")


def test_infer_country_from_vat() -> None:
    md = _master()
    sup = Supplier(vat_id="DE209177122")
    assert md.infer_country(sup) == "DE"


def test_parse_iso_date() -> None:
    assert parse_iso_date("2026-02-02") is not None
    assert parse_iso_date("02/02/2026") is None  # not ISO -> honest None
    assert parse_iso_date("") is None
