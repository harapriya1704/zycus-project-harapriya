"""Schema-shaped structuring chain (the "structured output" stage).

Takes the extracted text of one PDF and produces a :class:`src.schemas.FileOutput`
(payables + declined) whose shape is enforced by Pydantic. The LLM is told to
emit **raw** document values and to leave every master-data code ``""``; a
deterministic resolver (:mod:`src.master_data`) fills the codes afterwards so
no code is ever guessed.

Provider-agnostic output contract: the model is asked for plain JSON (no
provider-side strict tool schema), and the reply is validated against the
Pydantic schema *locally*. Fields the model invented that are not in the
schema are dropped (large open-weight models often add ``buyer.name`` etc.);
fields it omitted fall back to their schema defaults.
"""

from __future__ import annotations

import json
import re
import time
import typing
from functools import lru_cache

import pydantic
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.llm_retry import describe_error, invoke_with_retry
from src.logging_conf import get_logger
from src.master_data import MasterData
from src.schemas import Autodraft, Declined, FileOutput

log = get_logger(__name__)

#: Documented default for the precision-mode structuring input cap. Oversized
#: multi-page transcripts are NOT hard-cut from the front (which would drop the
#: last pages' line items and totals): :func:`_save_truncate` preserves both the
#: head (header, supplier, payment terms) and the tail (line items, duty and
#: charges, grand totals). Overridable via ``INV_MAX_STRUCTURING_CHARS``
#: (:attr:`Settings.max_structuring_chars`); ``0`` disables the cap.
#: Deterministic code resolution always sees the full text (``format_autodraft``
#: truncates only ``text_for_llm``, never ``document_text``).
MAX_STRUCTURING_CHARS = 12000

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
- quantity and unit_price are MANDATORY on every GOODS / SERVICE / FREIGHT
  line; the ERP prices a line as quantity x unit_price, so an empty price
  books the line as ZERO. Never leave them "" on a non-tax line.
    * When the document prints an explicit quantity and rate — e.g.
      "219744 KH * $0.0595/KH" or "475 kW X 4.77000" — set quantity to the raw
      count ("219744", "475") and unit_price to the rate ("0.0595", "4.77000"),
      dropping any unit label ("KH", "kW") and currency symbol.
    * For a flat-fee / lump-sum line with no printed quantity (e.g. "Customer
      Charge ... 299.78"), set quantity = "1" and unit_price equal to the line
      total exactly as printed. This default quantity "1" is a convention, not
      a value taken from the page.
- EXCLUDE category headers and subtotal / grand-total summaries from
  line_items: rows such as "Electric", "PECO ELECTRIC DELIVERY",
  "ELECTRIC SUPPLY", "TAXES & FEES", "Total New Charges", "Total Current
  Charges" are NOT atomic charges. Extract only atomic line charges — a single
  quantity x rate or a single flat fee — never a subtotal that aggregates
  other lines.
- item_type TAX lines carry their value in total / tax_amount; leave their
  quantity and unit_price "" (the ERP reads taxes from tax_amount / rate, not
  from qty x price).
- tax_amount may be "" — the ERP derives it from tax_rate. Use a negative
  tax_amount for withholding taxes that reduce what is owed.
- A credit is invoice_type "CREDIT_MEMO" with the same keys, positive values.
- NEVER fill master-data codes: supplier_id, buyer codes, payment_term_id,
  po_id, tax_type_code must all be "".
- item_type is one of GOODS | SERVICE | FREIGHT | TAX.
- A customs / duty / import document (often labelled "CUSTOMS_INVOICE", "DUTY
  INVOICE", "IMPORT DECLARATION") that CARRIES billable charges IS a payable:
  structure it. Import duty goes in excise_duties; brokerage, port/handling and
  clearance fees are SERVICE or FREIGHT line_items (quantity x unit_price or
  quantity "1" with unit_price = line total); freight goes in freight_charges;
  other one-off charges in extra_charges; VAT/import taxes in taxes[] or TAX
  lines. Decline such a document ONLY when it is a bare customs declaration
  with NO billable amounts (duty due absent, no fees, no taxes) — a declaration
  alone is not bookable.
- If the document is NOT a payable (e.g. a statement, advertisement, letter,
  estimate, quotation, unpaid dunning letter, duplicate, or otherwise
  not-bookable), return payables: [] and explain in declined[] with a real
  doc_type and reason.
- A single PDF may contain several payables; return one payable per bookable
  invoice. Pages that are not payables must not create payables."""


#: Streamlined system prompt for fast testing mode (``--fast``). Same hard
#: rules, but the verbose markdown walk-through and examples are dropped —
#: roughly half the input tokens of :data:`_STRUCTURING_SYSTEM_PROMPT`.
_STRUCTURING_SYSTEM_PROMPT_FAST = """You structure ONE supplier document's text into ERP autodraft JSON:
{"payables":[ ... ],"declined":[{"doc_type":"...","reason":"..."}]}

A payable (INVOICE or CREDIT_MEMO) has: invoice_number, invoice_date (ISO YYYY-MM-DD), due_date, currency, supplier{name, supplier_id:"", address, vat_id}, buyer{company_code:"", business_unit_code:"", location_code:""}, payment_term_id:"", po_number, po_id:"", gross_total, subtotal, total_tax_amount, discount_amount, freight_charges, insurance_charges, extra_charges, excise_duties, taxes[{tax_type, tax_name, tax_rate, tax_amount, tax_type_code:""}], line_items[{description,item_type,uom,quantity,unit_price,total,discount,discount_percentage,tax_rate,tax_amount,taxes[...]}].

HARD RULES:
- Reproduce values EXACTLY as printed; never compute totals the document does not print; a field not printed is "".
- Numbers are strings in dot-decimal form; convert European comma decimals ("1.234,56" -> "1234.56").
- quantity and unit_price are MANDATORY on non-tax lines; a flat-fee line uses quantity "1" and unit_price = line total.
- EXCLUDE category headers and subtotal/grand-total summaries from line_items (e.g. "Total New Charges", "TAXES & FEES"); keep only atomic charges.
- item_type is GOODS|SERVICE|FREIGHT|TAX. A TAX line carries total/tax_amount with empty quantity and unit_price.
- NEVER fill master-data codes: supplier_id, buyer codes, payment_term_id, po_id, tax_type_code stay "".
- A CUSTOMS / DUTY / IMPORT document that bills charges (import duty, brokerage, port/handling fees, freight, taxes) IS a payable: structure it. Put duty in excise_duties; brokerage/port/handling/clearance as SERVICE or FREIGHT line_items; freight in freight_charges; other one-off charges in extra_charges. Decline such a document ONLY when it carries no billable amounts (a bare customs declaration with no charges is not bookable).
- Not a payable (statement, duplicate, advert, estimate, quotation, letter, unpaid dunning letter) -> payables:[] and a declined[] entry with real doc_type/reason.
- A PDF may hold several payables. Return ONLY the JSON, no preamble."""


def _save_truncate(text: str, limit: int) -> str:
    """Truncate *text* to at most *limit* chars, preserving head AND tail.

    Multi-page invoices put the header, supplier and payment terms on page one
    and the line items, duty/freight charges and the grand total on the last
    pages. A naive ``text[:limit]`` drops that later financial context entirely
    — the exact failure that produced mis-classified declines. This keeps the
    first part of the transcript and the last part, joining them with an
    explicit marker so it is obvious the LLM does not see the full document.

    Args:
        text: the full document transcript.
        limit: maximum character budget for the window.

    Returns:
        ``text`` unchanged when it fits; otherwise a head+tail window of at
        most ``limit`` characters.
    """
    if len(text) <= limit:
        return text
    marker = f"\n... [MIDDLE OMITTED: {len(text) - limit} chars of this transcript are not shown] ...\n"
    tail_budget = min(len(text), limit // 2)
    head_budget = limit - len(marker) - tail_budget
    if head_budget <= 0:
        # Tiny limit: shrink the marker (and tail) rather than dropping the tail.
        marker = "\n... [MIDDLE OMITTED] ...\n"
        tail_budget = max(0, min(len(text), limit - len(marker) - 1))
        head_budget = max(0, limit - len(marker) - tail_budget)
    head = text[:head_budget]
    tail = text[len(text) - tail_budget :]
    return head + marker + tail


def build_autodraft_chain(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
    fast_mode: bool = False,
    system_prompt: str | None = None,
) -> Runnable:
    """Build the LangChain runnable: document text -> :class:`FileOutput`.

    The model is created by :meth:`Settings.get_text_llm` — a ``ChatOpenAI``
    wired to Groq (``GROQ_API_KEY``), so all text reasoning stays on Groq
    while image transcription runs on the Hugging Face Serverless router.

    The model is wired to the Pydantic :class:`FileOutput` schema via
    ``with_structured_output``, so the response is validated strictly (no
    drifting keys or wrong types) before any downstream code sees it.

    Args:
        model: chat model name; defaults to ``INV_TEXT_MODEL`` (via
            :meth:`Settings.effective_structuring_model`).
        temperature: sampling temperature; defaults to the configured value.
        settings: settings source; defaults to the process singleton.
        fast_mode: use the streamlined fast-mode system prompt (half the
            input tokens, same hard rules).
        system_prompt: override the system prompt entirely (used by the
            customs/duty re-structure fallback, which appends a corrective
            instruction to the stock prompt).

    Returns:
        A ``Runnable`` accepting ``{"document_text": str}`` and returning a
        validated :class:`FileOutput`.

    Raises:
        RuntimeError: when no Groq API key is configured.
    """
    cfg = settings or get_settings()
    model_name = model or cfg.effective_structuring_model()
    temp = cfg.structuring_temperature if temperature is None else temperature

    try:
        llm = cfg.get_text_llm(model=model_name, temperature=temp)
    except Exception as exc:  # pragma: no cover - credentials are checked up-front
        log.error("ChatOpenAI construction failed: %s (%s)", type(exc).__name__, exc)
        raise RuntimeError(
            "Cannot build the structuring chain: no usable Groq credentials "
            "configured (set GROQ_API_KEY)."
        ) from exc

    # The system prompt is passed as a *fixed* SystemMessage (not a template):
    # it contains literal ``{``/``}`` for the JSON examples, which must not be
    # treated as f-string replacement fields. Fast mode swaps in the compact
    # variant (identical rules, far fewer input tokens); an explicit override
    # always wins over either stock prompt.
    if system_prompt is not None:
        base_prompt = system_prompt
    else:
        base_prompt = _STRUCTURING_SYSTEM_PROMPT_FAST if fast_mode else _STRUCTURING_SYSTEM_PROMPT
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=base_prompt),
            ("human", "Document text:\n\n{document_text}"),
        ]
    )

    chain: Runnable = prompt | _ask(llm, settings=settings) | _finalize(FileOutput)
    return chain


#: ``doc_type`` values that mean "customs / duty / import document". These ARE
#: bookable when they carry duty/brokerage/port/freight/tax amounts — see
#: :func:`_looks_like_billable_customs` and the re-structure fallback in
#: :func:`format_autodraft`.
_CUSTOMS_DECLINE_TYPES: frozenset[str] = frozenset(
    {
        "CUSTOMS_INVOICE",
        "CUSTOMS",
        "CUSTOMS_DECLARATION",
        "CUSTOMS_DUTY_INVOICE",
        "DUTY_INVOICE",
        "DUTY",
        "IMPORT_INVOICE",
        "IMPORT",
        "IMPORT_DECLARATION",
        "HARMONIZED",
    }
)

#: Keywords that mark a customs/duty/import context.
_CUSTOMS_KEYWORDS: tuple[str, ...] = (
    "customs",
    "custom",
    "duty",
    "duties",
    "import",
    "brokerage",
    "broker",
    "declaration",
    "harmonized",
    "harmonised",
    "hs code",
    "clearance",
)

#: Currency/code-prefixed or dot/comma-decimal money markers. Used to decide
#: whether a customs-looking decline actually carried billable amounts.
_AMOUNT_PATTERN = re.compile(
    r"(?:\u20ac|USD\s*\$?|EUR|GBP|INR|\$\s*)?(?<![A-Za-z])\d[\d.,]{1,}\.\d{2}(?![A-Za-z])"
    r"|(?:\u20ac|\$|UK\u00a3|UK ?\u00a3|\u00a3|\bEUR\b|\bUSD\b|\bGBP\b)\s?\d[\d.,]{0,}",
    re.IGNORECASE,
)

#: Instruction appended to the stock system prompt when a customs/duty decline
#: is detected on a document that visibly carries billable amounts. One bounded
#: re-structure restores the payable instead of silently dropping it.
_CUSTOMS_RESTRUCTURE_NOTE = """

IMPORTANT CORRECTION:
The previous pass classified this document as a customs/duty/import document
and declined it, but the document text carries BILLABLE amounts (import duty,
brokerage, port/handling/clearance fees, freight and/or taxes). Re-classify it
as a PAYABLE and structure it: put import duty in "excise_duties";
brokerage/port/handling/clearance charges as SERVICE or FREIGHT line_items
(with explicit quantity and unit_price; a flat fee uses quantity "1" and
unit_price equal to the line total); freight in "freight_charges"; other
one-off charges in "extra_charges"; VAT/import taxes in "taxes". Do NOT decline
the document again unless it genuinely shows no billable amounts."""


def _looks_like_billable_customs(text: str) -> bool:
    """True when *text* mentions customs/duty/import AND carries money amounts.

    A bare customs declaration (no duty, fees or taxes) is legitimately not
    bookable; a customs/duty invoice that prints duties, brokerage or freight is
    billable and must be structured, not declined.
    """
    lowered = text.lower()
    if not any(keyword in lowered for keyword in _CUSTOMS_KEYWORDS):
        return False
    return bool(_AMOUNT_PATTERN.search(text))


def _ask(llm: ChatOpenAI, settings: Settings | None = None) -> Runnable:
    """Runnable mapping a message list to the model's (string) reply.

    Retry behaviour (budget and per-wait ceiling) follows the caller's
    :class:`Settings` rather than the module defaults, so a benchmark can force
    ``max_attempts=1`` / a 60 s wait cap and fail fast on a throttled provider.
    """

    def _invoke(messages) -> str:
        cfg = settings or get_settings()
        try:
            response = invoke_with_retry(
                llm,
                messages,
                max_attempts=cfg.llm_retry_attempts,
                max_wait=cfg.llm_max_wait,
            )
        except Exception as exc:  # noqa: BLE001 - surface the raw provider error
            log.error("Structuring LLM call failed: %s", describe_error(exc))
            raise
        return str(response.content).strip()

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(_invoke, name="llm_chat")


def _extract_json(text: str) -> str:
    """Extract the outermost ``{...}`` JSON object from a model reply.

    Models sometimes wrap JSON in markdown fences or add prose; providers may
    also truncate replies. This is defensive: it returns the largest balanced
    JSON-ish substring, or the original text unchanged if none is found.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return cleaned[start:]


def _strip_extra(
    data: object,
    schema: type[pydantic.BaseModel],
) -> object:
    """Recursively keep only fields declared by ``schema``.

    open-weight models (per the Groq target) frequently invent descriptive
    fields the contract does not allow (e.g. ``buyer.name``, ``buyer.address``).
    The schema keeps ``extra="forbid"`` for our own writes, so the *parser*
    boundary is where unknown keys are dropped — deterministically, following
    the schema itself.
    """

    def _clean(value: object, ann: object) -> object:
        if isinstance(value, list):
            item_anns = typing.get_args(ann) if typing.get_origin(ann) is list else (ann,)
            item_ann = item_anns[0] if item_anns else ann
            return [_clean(v, item_ann) for v in value]
        if (
            isinstance(value, dict)
            and isinstance(ann, type)
            and issubclass(ann, pydantic.BaseModel)
        ):
            keep = {}
            for key, item in value.items():
                if key not in ann.model_fields:
                    log.debug("Dropping undeclared field %s.%s", ann.__name__, key)
                    continue
                keep[key] = _clean(item, ann.model_fields[key].annotation)
            return keep
        return value

    return _clean(data, schema)


def _finalize(schema: type[pydantic.BaseModel]) -> Runnable:
    """Parse+validate the model's JSON string against a Pydantic schema.

    Provider-agnostic: instead of relying on ``with_structured_output``'s
    server-side tool schema (which some OpenAI-compatible providers — including
    Groq — enforce so strictly that a model omitting a defaulted field fails),
    we validate the response locally. Missing fields fall back to their schema
    defaults; extra fields the model invented are dropped (see ``_strip_extra``).
    """

    def _parse(text: str) -> pydantic.BaseModel:
        payload = _extract_json(text)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            log.error("Model JSON failed to decode (%s); response: %s", exc, payload[:500])
            raise
        data = _strip_extra(data, schema)
        try:
            return schema.model_validate(data)
        except pydantic.ValidationError as exc:
            log.error("Model JSON failed validation (%s); response: %s", exc, payload[:500])
            raise

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(_parse, name=f"{schema.__name__}_parser")


@lru_cache(maxsize=8)
def _chain_for(model: str, temperature: float, fast_mode: bool) -> Runnable:
    """Cached structuring chain per (model, temperature, fast mode)."""
    return build_autodraft_chain(model=model, temperature=temperature, fast_mode=fast_mode)


def _structuring_chain(settings: Settings | None = None) -> Runnable:
    """Resolve the structuring chain for the given settings.

    Fast mode selects the lightweight model and the compact prompt; precision
    mode uses the configured heavy model and the full system prompt.
    """
    cfg = settings or get_settings()
    return _chain_for(cfg.effective_structuring_model(), cfg.structuring_temperature, cfg.fast_mode)


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
    struct_model = cfg.effective_structuring_model() if chain is None else "override-chain"

    # Token budget: fast mode caps the input at ``fast_max_input_chars``
    # (default 3000). Precision mode applies ``max_structuring_chars`` (default
    # 12000; 0 = unlimited) with a head+tail split so the last pages' line
    # items, charges and totals are never dropped. Deterministic code
    # resolution still sees the full text, so master matching is unaffected.
    cap = cfg.effective_max_input_chars()
    text_for_llm = document_text if cap <= 0 else document_text[:cap]
    limit = cfg.max_structuring_chars
    if limit > 0 and len(text_for_llm) > limit:
        text_for_llm = _save_truncate(text_for_llm, limit)
        log.info(
            "%s: truncated structuring input to %s chars (was %s, head+tail kept)",
            filename,
            limit,
            len(document_text),
        )
    log.info(
        "%s: structuring %s chars of text (budget %s)",
        filename,
        len(text_for_llm),
        cap or limit or "unlimited",
    )
    _struct_start = time.perf_counter()
    result: FileOutput = effective_chain.invoke({"document_text": text_for_llm})
    log.info(
        "[Text Groq] Schema formatted in %.2fs (model %s)",
        time.perf_counter() - _struct_start,
        struct_model,
    )

    # Customs/duty mis-classification backstop: a billable customs/duty invoice
    # (duty, brokerage, freight, taxes present) must never be silently declined.
    # If the first pass declined it and the document visibly carries amounts,
    # re-structure once with an explicit corrective instruction. Bounded to one
    # extra call; a retry failure falls back to the original decline.
    if not result.payables and _looks_like_billable_customs(document_text):
        declined_types = {str(d.doc_type).strip().upper() for d in result.declined if d.doc_type}
        if declined_types & _CUSTOMS_DECLINE_TYPES:
            base_prompt = (
                _STRUCTURING_SYSTEM_PROMPT_FAST
                if cfg.fast_mode
                else _STRUCTURING_SYSTEM_PROMPT
            )
            retry_chain = build_autodraft_chain(
                temperature=cfg.structuring_temperature,
                settings=cfg,
                fast_mode=cfg.fast_mode,
                system_prompt=base_prompt + _CUSTOMS_RESTRUCTURE_NOTE,
            )
            try:
                retried = retry_chain.invoke({"document_text": text_for_llm})
            except Exception as exc:  # noqa: BLE001 - keep the original decline on failure
                log.warning(
                    "%s: customs/duty re-structure failed (%s); keeping the first decline",
                    filename,
                    type(exc).__name__,
                )
                retried = None
            if retried is not None and retried.payables:
                log.info(
                    "%s: declined as customs/duty but billable amounts found; "
                    "re-structured -> %s payable(s)",
                    filename,
                    len(retried.payables),
                )
                result = retried

    result.file = filename

    effective_master = master if master is not None else MasterData.load_default(settings=cfg)
    return resolve_codes(result, effective_master, document_text=document_text)
