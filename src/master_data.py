"""Master-data loading and deterministic code resolution.

The ERP schema demands master-data **codes** wherever a match exists
(``supplier_id``, buyer codes, ``tax_type_code``, ``payment_term_id``,
``po_id``) and an honest ``""`` otherwise. The LLM never guesses these — it
extracts raw document values and this module resolves the codes.

Matching is built for scale the way the brief insists: masters are indexed
once into normalised dicts, so each lookup is O(1) regardless of master size,
and nothing is hardcoded to the few sample rows.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.logging_conf import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> str:
    """Casefold + strip diacritics + collapse whitespace for fuzzy matching."""
    if value is None:
        return ""
    s = str(value).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


def _norm_compact(value: Any) -> str:
    """Like :func:`_norm` but keeps only alphanumerics (for id-like matching)."""
    return "".join(ch for ch in _norm(value) if ch.isalnum())


def _norm_number(value: Any) -> str:
    """Normalise a numeric-looking string (drops separators/currency/%)."""
    if value is None:
        return ""
    s = str(value).strip().replace("%", "").replace(",", "").replace(" ", "")
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    return match.group(0) if match else s


def parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO-ish ``YYYY-MM-DD`` date (tolerates '/' separators)."""
    if not value:
        return None
    clean = _norm(value).replace("/", "-").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean):
        return None
    try:
        return datetime.strptime(clean, "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any]:
    """Load a master JSON file, tolerating a missing file (returns empty dict)."""
    if not path.is_file():
        log.warning("Master data file not found: %s", path)
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------

@dataclass
class MasterData:
    """Indexed reference data used to resolve ERP codes.

    Build once per process (:meth:`MasterData.load`), then call the
    ``match_*`` / ``resolve_*`` methods per document. All lookups hit
    pre-computed dict indexes, keeping matching cheap at master scale.
    """

    suppliers: list[dict[str, str]] = field(default_factory=list)
    companies: list[dict[str, Any]] = field(default_factory=list)
    taxes: list[dict[str, str]] = field(default_factory=list)
    payment_terms: list[dict[str, Any]] = field(default_factory=list)
    purchase_orders: list[dict[str, Any]] = field(default_factory=list)

    # Indexes
    _by_vat: dict[str, dict[str, str]] = field(default_factory=dict)
    _by_name: dict[str, dict[str, str]] = field(default_factory=dict)
    _tax_by_key: dict[tuple, list[dict[str, str]]] = field(default_factory=dict)
    _term_by_alias: dict[str, str] = field(default_factory=dict)
    _po_by_number: dict[str, str] = field(default_factory=dict)
    _bu_by_name: dict[str, tuple[str, str]] = field(default_factory=dict)
    _loc_by_address: dict[str, tuple[str, str, str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, master_dir: Path) -> MasterData:
        """Load and index all masters under ``master_dir``."""
        md = cls()
        md._load_suppliers(_load_json(master_dir / "suppliers.json").get("suppliers", []))
        md._load_companies(_load_json(master_dir / "chart_of_books.json").get("companies", []))
        md._load_taxes(_load_json(master_dir / "tax_master.json").get("taxes", []))
        md._load_payment_terms(_load_json(master_dir / "payment_terms.json").get("payment_terms", []))
        md._load_pos(_load_json(master_dir / "po_master.json").get("purchase_orders", []))
        log.info("Master data loaded: %s suppliers, %s taxes, %s POs, %s terms",
                 len(md.suppliers), len(md.taxes), len(md.purchase_orders), len(md.payment_terms))
        return md

    @classmethod
    def load_default(cls, settings: Settings | None = None) -> MasterData:
        """Load from the configured master-data directory."""
        cfg = settings or get_settings()
        return cls.load(cfg.resolved_master_data_dir())

    # ------------------------------------------------------------------
    def _load_suppliers(self, rows: list[dict]) -> None:
        self.suppliers = rows
        for row in rows:
            sid = str(row.get("supplier_id", ""))
            if not sid:
                continue
            self._by_name[_norm(row.get("name"))] = row
            for key in ("vat_id", "bank_iban", "email"):
                if row.get(key):
                    self._by_vat[_norm_compact(row.get(key))] = row

    def _load_companies(self, companies: list[dict]) -> None:
        self.companies = companies
        for company in companies:
            for bu in company.get("business_units", []):
                self._bu_by_name[_norm(bu.get("business_unit_name"))] = (
                    str(company.get("company_code", "")),
                    str(bu.get("business_unit_code", "")),
                )
                for loc in bu.get("locations", []):
                    self._loc_by_address[_norm(loc.get("invoice_to_address"))] = (
                        str(company.get("company_code", "")),
                        str(bu.get("business_unit_code", "")),
                        str(loc.get("location_code", "")),
                    )

    def _load_taxes(self, rows: list[dict]) -> None:
        self.taxes = rows
        for row in rows:
            key = (
                _norm(row.get("tax_type")),
                _norm_number(row.get("rate")),
                _norm(row.get("name")),
            )
            self._tax_by_key.setdefault(key, []).append(row)

    def _load_payment_terms(self, rows: list[dict]) -> None:
        self.payment_terms = rows
        for row in rows:
            for alias in row.get("text_aliases", []):
                self._term_by_alias[_norm(alias)] = str(row.get("payment_term_id", ""))

    def _load_pos(self, rows: list[dict]) -> None:
        self.purchase_orders = rows
        for row in rows:
            self._po_by_number[_norm_compact(row.get("po_number"))] = str(row.get("po_id", ""))

    # ------------------------------------------------------------------
    # Resolvers
    # ------------------------------------------------------------------
    def match_supplier(self, *, name: str = "", vat_id: str = "", address: str = "") -> str:
        """Return a ``supplier_id`` or ``""`` (industry-ranked heuristics)."""
        if vat_id:
            hit = self._by_vat.get(_norm_compact(vat_id))
            if hit:
                return str(hit["supplier_id"])
        if name:
            hit = self._by_name.get(_norm(name))
            if hit:
                return str(hit["supplier_id"])
        if name:
            nname = _norm(name)
            naddr = _norm(address)
            # Name token match — one side contains all significant tokens of the other.
            for row in self.suppliers:
                if nname and _all_tokens(nname, _norm(row.get("name"))):
                    return str(row["supplier_id"])
                if naddr and row.get("address") and _all_tokens(naddr, _norm(row.get("address"))):
                    return str(row["supplier_id"])
        return ""

    def match_buyer(self, *, name: str = "", address: str = "", text: str = "") -> tuple[str, str, str]:
        """Return (company_code, business_unit_code, location_code); empties on no match.

        The schema carries no buyer-address field, so callers may pass the full
        document ``text``: any indexed business-unit name or location address
        found inside it wins.
        """
        nname, naddr = _norm(name), _norm(address)
        if nname and nname in self._bu_by_name:
            company, bu = self._bu_by_name[nname]
            loc = ""
            for _addr, (c2, b2, l2) in self._loc_by_address.items():
                if c2 == company and b2 == bu:
                    loc = l2
                    break
            return company, bu, loc
        if naddr:
            for addr, (company, bu, loc) in self._loc_by_address.items():
                if naddr in addr or addr in naddr or _all_tokens(naddr, addr):
                    return company, bu, loc
        if text:
            found = _norm(text)
            best: tuple[str, str, str] = ("", "", "")
            for addr, codes in self._loc_by_address.items():
                if addr in found or _all_tokens(addr, found):
                    best = codes
                    break
            if any(best):
                return best
            for bu_name, codes in self._bu_by_name.items():
                if bu_name in found:
                    company, bu_code = codes
                    loc = next(
                        (l2 for addr2, (c2, b2, l2) in self._loc_by_address.items()
                         if c2 == company and b2 == bu_code),
                        "",
                    )
                    return company, bu_code, loc
        return "", "", ""

    def match_tax(self, *, tax_type: str = "", tax_name: str = "", rate: str = "", country: str = "") -> str:
        """Return a ``tax_type_code`` or ``""``."""
        hits: list[dict[str, str]] = []
        if rate:
            hits = self._tax_by_key.get((_norm(tax_type), _norm_number(rate), _norm(tax_name)), [])
            if not hits:
                # rate+type without an exact name match
                hits = [r for r in self.taxes
                        if _norm_number(r.get("rate")) == _norm_number(rate) and _norm(r.get("tax_type")) == _norm(tax_type)]
        if not hits and tax_name:
            hits = [r for r in self.taxes if _all_tokens(_norm(tax_name), _norm(r.get("name")))]
        return self._pick_tax(hits, country)

    @staticmethod
    def _pick_tax(hits: list[dict[str, str]], country: str) -> str:
        if not hits:
            return ""
        for hit in hits:
            if country and _norm(hit.get("country")) == _norm(country):
                return str(hit.get("code", ""))
        return str(hits[0].get("code", ""))

    def match_payment_term(self, *, text: str = "", invoice_date: str = "", due_date: str = "") -> str:
        """Resolve via alias text first; else by the invoice->due day span."""
        if text:
            ntext = _norm(text)
            for alias, term_id in self._term_by_alias.items():
                if alias in ntext or ntext in alias:
                    return term_id
        inv = parse_iso_date(invoice_date)
        due = parse_iso_date(due_date)
        if inv and due:
            days = (due - inv).days
            nearest = min(self.payment_terms, key=lambda t: abs(int(t.get("days", 0)) - days))
            return str(nearest.get("payment_term_id", ""))
        return ""

    def match_po(self, po_number: str) -> str:
        """Return the matched ``po_id`` or ``""``."""
        return self._po_by_number.get(_norm_compact(po_number), "")

    # ------------------------------------------------------------------
    def infer_country(self, supplier: Any) -> str:
        """Best-effort supplier country used to disambiguate tax codes."""
        from src.schemas import Supplier

        sup: Supplier = supplier
        national_id = (sup.vat_id or "").upper()
        eu_prefix = {"DE", "EE", "GB", "PT", "PL", "DK", "RO", "SE"}
        for prefix in eu_prefix:
            if national_id.startswith(prefix):
                return prefix
        return ""


def _all_tokens(a: str, b: str) -> bool:
    """True when one token set is fully contained in the other (fuzzy match)."""
    if not a or not b:
        return False
    ta, tb = set(a.split()), set(b.split())
    return ta <= tb or tb <= ta
