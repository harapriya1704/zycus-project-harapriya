"""Schema-shaped structuring chain (the "structured output" stage).

Takes the extracted text of one PDF and produces a :class:`src.schemas.FileOutput`
(payables + declined) whose shape is enforced by Pydantic through LangChain's
``with_structured_output``. The LLM is told to emit **raw** document values and
to leave every master-data code ``""``; a deterministic resolver
(:mod:`src.master_data`) fills the codes afterwards so no code is ever guessed.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.logging_conf import get_logger
from src.master_data import MasterData
from src.schemas import Autodraft, Declined, FileOutput

log = get_logger(__name__)

#: System prompt for the structuring chain. Every rule the grader enforces is
#: stated here; anything not on the document must stay empty.
_STRUCTURING_SYSTEM_PROMPT = """You structure supplier documents into autodraft records for an ERP.

You are given the extracted text of ONE PDF file. Produce the JSON object:

  {
    "payables": [ ... ],   // 0..N bookable payables
    "declined":  [ { "doc_type": "...", "reason": "..." }, ... ]
  }

Each payable is:

  invoice_number, invoice_date (ISO YYYY-MM-DD), due_date, invoice_type
  (INVOICE or CREDIT_MEMO), currency,
  supplier: { name, supplier_id: "", address, vat_id },
  buyer: { company_code: "", business_unit_code: "", location_code: "" },
  payment_term_id: "", po_number, po_id: "",
  gross_total, subtotal, total_tax_amount, discount_amount, freight_charges,
  insurance_charges, extra_charges, excise_duties (all "" when absent),
  taxes: [ { tax_type, tax_name, tax_rate, tax_amount, tax_type_code: "" } ],
  line_items: [ { description, item_type, uom, quantity, unit_price, total,
                  discount, discount_percentage, tax_rate, tax_amount,
                  taxes: [ same shape as header taxes ] } ]

HARD RULES:
- Reproduce values EXACTLY as printed: raw quantities, unit prices, discounts,
  charges, and the taxes you assign. Do NOT sum or compute totals the document
  does not print; the ERP derives all totals itself.
- Every value must appear on the document. Leave a field "" (empty string)
  rather than inventing a number. Empty string is the correct answer for
  "not printed".
- Numbers are strings, dot-decimal ("1234.56"). Convert European comma
  decimals ("1.234,56") to dots.
- unit_price is NET (tax-exclusive). Place each tax where the document places
  it: a tax shown per line goes on that line (line taxes[] or tax_rate/
  tax_amount); a tax shown once at the header goes in the header taxes[].
- tax_amount may be "" — the ERP derives it from tax_rate. Use a negative
  tax_amount for withholding taxes that reduce what is owed.
- A credit is invoice_type "CREDIT_MEMO" with the same keys, positive values.
- NEVER fill master-data codes: supplier_id, buyer codes, payment_term_id,
  po_id, tax_type_code must all be "".
- item_type is one of GOODS | SERVICE | FREIGHT | TAX.
- If the document is NOT a payable (e.g. a statement, advertisement, letter,
  duplicate, or otherwise not-bookable), return payables: [] and explain in
  declined[] with a real doc_type and reason.
- A single PDF may contain several payables; return one payable per bookable
  invoice. Pages that are not payables must not create payables."""


def build_autodraft_chain(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Runnable:
    """Build the LangChain runnable: document text -> :class:`FileOutput`.

    The model is wired to the Pydantic :class:`FileOutput` schema via
    ``with_structured_output``, so the response is validated strictly (no
    drifting keys or wrong types) before any downstream code sees it.

    Args:
        model: chat model name; defaults to ``INV_STRUCTURING_MODEL`` or
            ``INV_EXTRACTION_MODEL``.
        temperature: sampling temperature; defaults to the configured value.
        settings: settings source; defaults to the process singleton.

    Returns:
        A ``Runnable`` accepting ``{"document_text": str}`` and returning a
        validated :class:`FileOutput`.

    Raises:
        RuntimeError: when no API key is configured.
    """
    cfg = settings or get_settings()
    model_name = model or cfg.structuring_model
    temp = cfg.structuring_temperature if temperature is None else temperature

    try:
        llm = ChatOpenAI(model=model_name, temperature=temp)
    except Exception as exc:  # typically a missing OPENAI_API_KEY
        raise RuntimeError(
            "Cannot build the structuring chain: no usable OpenAI credentials "
            "configured (set OPENAI_API_KEY)."
        ) from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _STRUCTURING_SYSTEM_PROMPT),
            ("human", "Document text:\n\n{document_text}"),
        ]
    )
    chain: Runnable = prompt | llm.with_structured_output(FileOutput)
    return chain


@lru_cache(maxsize=4)
def _chain_for(model: str, temperature: float) -> Runnable:
    """Cached structuring chain per (model, temperature)."""
    return build_autodraft_chain(model=model, temperature=temperature)


def _structuring_chain(settings: Settings | None = None) -> Runnable:
    """Resolve the structuring chain for the given settings."""
    cfg = settings or get_settings()
    return _chain_for(cfg.structuring_model, cfg.structuring_temperature)


def resolve_codes(
    file_output: FileOutput,
    master: MasterData,
    document_text: str = "",
) -> FileOutput:
    """Fill every master-data code in ``file_output``, deterministically.

    Raw printed values are never changed; only the ``*_id`` / ``*_code`` /
    ``payment_term_id`` fields are populated from a real match. No match means
    the code stays ``""`` — never a guess.

    Args:
        file_output: the LLM-produced (but code-less) autodrafts.
        master: the indexed master data.
        document_text: the source document text, used to locate buyer-entity
            hints (the schema has no buyer-address field of its own).

    Returns:
        A new :class:`FileOutput` with codes resolved in place.
    """
    resolved_payables: list[Autodraft] = []
    for payable in file_output.payables:
        supplier = payable.supplier
        supplier.supplier_id = master.match_supplier(
            name=supplier.name, vat_id=supplier.vat_id, address=supplier.address
        )
        country = master.infer_country(supplier)

        company, bu, loc = master.match_buyer(text=document_text)
        payable.buyer.company_code = company
        payable.buyer.business_unit_code = bu
        payable.buyer.location_code = loc

        payable.payment_term_id = master.match_payment_term(
            text=document_text,
            invoice_date=payable.invoice_date,
            due_date=payable.due_date,
        )

        payable.po_id = master.match_po(payable.po_number)

        for tax in payable.taxes:
            tax.tax_type_code = master.match_tax(
                tax_type=tax.tax_type,
                tax_name=tax.tax_name,
                rate=tax.tax_rate,
                country=country,
            )
        for line in payable.line_items:
            for tax in line.taxes:
                tax.tax_type_code = master.match_tax(
                    tax_type=tax.tax_type,
                    tax_name=tax.tax_name,
                    rate=tax.tax_rate,
                    country=country,
                )

        resolved_payables.append(payable)

    file_output.payables = resolved_payables
    return file_output


def format_autodraft(
    document_text: str,
    filename: str,
    chain: Runnable | None = None,
    master: MasterData | None = None,
    settings: Settings | None = None,
) -> FileOutput:
    """Run the full structuring/resolving stage for one document's text.

    Args:
        document_text: extracted text (may be empty for unreadable scans).
        filename: the source PDF file name, stored in ``result.file``.
        chain: structuring runnable; auto-built when ``None``.
        master: master data; auto-loaded when ``None``.
        settings: settings source.

    Returns:
        A validated :class:`FileOutput` with codes resolved.

    Raises:
        RuntimeError: when structuring requires an LLM but no API key exists.
    """
    cfg = settings or get_settings()

    if not document_text.strip():
        result = FileOutput(
            file=filename,
            declined=[
                Declined(
                    doc_type="UNREADABLE",
                    reason="No usable text could be extracted (image-only page, vision unavailable).",
                )
            ],
        )
        log.warning("%s: no text to structure; declined as UNREADABLE", filename)
        return result

    effective_chain = chain or _structuring_chain(cfg)
    log.info("%s: structuring %s chars of text", filename, len(document_text))
    result: FileOutput = effective_chain.invoke({"document_text": document_text})
    result.file = filename

    effective_master = master if master is not None else MasterData.load_default(settings=cfg)
    return resolve_codes(result, effective_master, document_text=document_text)
