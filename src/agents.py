"""Multi-agent validation loop (day 2).

Two agents cooperate on every document, orchestrated by a bounded loop:

1. **Proposer agent** — the day-1 structuring chain (:mod:`src.formatter`):
   document text -> :class:`FileOutput` with master-data codes resolved.
2. **Validation agent** — deterministic (:mod:`src.validation`): cross-checks
   the proposer's output against the master-data files and the ERP oracle
   (:mod:`erp`), returning flagged inconsistencies.
3. **Corrector agent** — an LLM that receives the proposer's JSON *plus* the
   validation issues (and the raw document text) and emits a revised
   :class:`Autodraft`, re-validated in the next iteration.

The loop is hard-capped by ``max_retries`` (configured via ``INV_MAX_RETRIES``,
default **3**) so it terminates in finite time whatever the model does; the
last produced autodraft (and its remaining issues) is always returned.

All LLM agents target the same OpenAI-compatible provider configured in
``.env`` (Groq: ``llama-3.3-70b-versatile``) via :class:`src.config.Settings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from src.config import Settings, get_settings
from src.formatter import _ask, _finalize, format_autodraft, resolve_codes
from src.logging_conf import get_logger
from src.master_data import MasterData
from src.schemas import Autodraft, FileOutput
from src.validation import MasterDataValidator, Severity, ValidationIssue

log = get_logger(__name__)

#: System prompt for the corrector agent. It must repair *only* the flagged
#: fields and must never invent values that are not on the document.
_CORRECTOR_SYSTEM_PROMPT = """You are the Corrector agent in an invoice-processing pipeline.

You receive:
- "proposed": a JSON autodraft produced by a proposer agent (codes already resolved);
- "issues": validation issues cross-referenced against master data and the ERP oracle;
- "document_text": the raw text of the source document.

Your job: return a corrected autodraft (same JSON shape) that resolves as many
of the flagged inconsistencies as possible. Rules:
- Only change a field when an issue requires it. Leave correct fields untouched.
- Reproduce values EXACTLY as printed on the document; never invent numbers.
- Numbers are strings in dot-decimal form ("1234.56").
- Master-data codes (supplier_id, buyer codes, payment_term_id, po_id,
  tax_type_code) stay empty unless you are fixing an explicitly flagged,
  verifiable mismatch. Do not guess codes.
- If an issue cannot be resolved from the document, leave the field unchanged
  rather than guessing.
- Return ONLY the corrected autodraft JSON, with no preamble."""


@dataclass
class ValidationResult:
    """Outcome of the validation loop for one document."""

    file_output: FileOutput
    issues: list[ValidationIssue] = field(default_factory=list)
    attempts: int = 0  # corrections actually issued
    converged: bool = False  # True when the final proposal is error-free
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Corrector chain
# ---------------------------------------------------------------------------

def build_corrector_chain(
    model: str | None = None,
    temperature: float | None = None,
    settings: Settings | None = None,
) -> Runnable:
    """Build the corrector runnable: (proposed, issues, text) -> :class:`Autodraft`.

    Args:
        model: chat model name; defaults to ``INV_STRUCTURING_MODEL``.
        temperature: sampling temperature; defaults to the configured value.
        settings: settings source; defaults to the process singleton.

    Returns:
        A ``Runnable`` accepting a dict with ``proposed_json``, ``issues_json``
        and ``document_text`` keys and returning a validated :class:`Autodraft`.

    Raises:
        RuntimeError: when no API key is configured.
    """
    cfg = settings or get_settings()
    model_name = model or cfg.structuring_model
    temp = cfg.structuring_temperature if temperature is None else temperature

    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=temp,
            base_url=cfg.provider_base_url(),
            api_key=cfg.provider_api_key(),
            max_tokens=cfg.llm_max_tokens,
        )
    except Exception as exc:  # typically a missing OPENAI_API_KEY
        log.error("ChatOpenAI (corrector) construction failed: %s (%s)", type(exc).__name__, exc)
        raise RuntimeError(
            "Cannot build the correction chain: no usable OpenAI-compatible credentials "
            "configured (set OPENAI_API_KEY / OPENAI_BASE_URL)."
        ) from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=_CORRECTOR_SYSTEM_PROMPT),
            (
                "human",
                "Proposed autodraft:\n\n{proposed_json}\n\n"
                "Validation issues:\n\n{issues_json}\n\n"
                "Document text:\n\n{document_text}",
            ),
        ]
    )
    chain: Runnable = prompt | _ask(llm) | _finalize(Autodraft)
    return chain


@lru_cache(maxsize=4)
def _corrector_for(model: str, temperature: float) -> Runnable:
    """Cached corrector chain per (model, temperature)."""
    return build_corrector_chain(model=model, temperature=temperature)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class ValidationLoop:
    """Orchestrates proposer/validator/corrector with a bounded retry count.

    Args:
        master: indexed master data.
        settings: settings source; defaults to the process singleton.
        max_retries: correction iterations allowed (default 3). This caps the
            loop so it can never run indefinitely regardless of model behaviour.
        structuring_chain: proposer runnable override (tests).
        corrector_chain: corrector runnable override (tests).
    """

    def __init__(
        self,
        *,
        master: MasterData,
        settings: Settings | None = None,
        max_retries: int | None = None,
        structuring_chain: Runnable | None = None,
        corrector_chain: Runnable | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.master = master
        self.max_retries = max_retries if max_retries is not None else cfg.max_retries
        self.structuring_chain = structuring_chain
        self.corrector_chain = corrector_chain
        self.validator = MasterDataValidator(master)
        self.stats: dict[str, int] = {"converged": 0, "not_converged": 0, "corrections": 0}

    # ------------------------------------------------------------------
    def run(self, document_text: str, filename: str) -> ValidationResult:
        """Run the full propose -> validate -> correct loop for one document.

        Args:
            document_text: extracted text of the source document.
            filename: the source PDF file name (stored in ``result.file``).

        Returns:
            A :class:`ValidationResult` with the final (possibly un-converged)
            file output, the remaining issues and how many corrections ran.
        """
        file_output = self._propose(document_text, filename)
        issues = self.validator.validate(file_output, document_text=document_text)

        attempts = 0
        while self._has_errors(issues) and attempts < self.max_retries:
            file_output = self._correct(file_output, issues, document_text, filename)
            issues = self.validator.validate(file_output, document_text=document_text)
            attempts += 1
            log.info("%s: correction attempt %s/%s, %s errors remaining",
                     filename, attempts, self.max_retries, self._error_count(issues))

        converged = not self._has_errors(issues)
        self.stats["corrections"] += attempts
        self.stats["converged" if converged else "not_converged"] += 1
        log.info("%s: validation %s after %s corrections (%s issues)",
                 filename, "CONVERGED" if converged else "NOT CONVERGED", attempts, len(issues))
        return ValidationResult(
            file_output=file_output,
            issues=issues,
            attempts=attempts,
            converged=converged,
            max_retries=self.max_retries,
        )

    # ------------------------------------------------------------------
    def _propose(self, document_text: str, filename: str) -> FileOutput:
        """Agent 1 (proposer): structure + resolve codes, or honest decline."""
        return format_autodraft(
            document_text,
            filename=filename,
            chain=self.structuring_chain,
            master=self.master,
            settings=self.settings,
        )

    def _correct(
        self,
        proposed: FileOutput,
        issues: list[ValidationIssue],
        document_text: str,
        filename: str,
    ) -> FileOutput:
        """Agent 3 (corrector): ask the LLM to repair every payable's issues.

        Non-error issues are advisory and passed along as extra context; the
        corrector may choose to act on them. Codes are re-resolved afterwards.
        """
        corrected_payables: list[Autodraft] = []
        for idx, payable in enumerate(proposed.payables):
            payable_issues = [i for i in issues if i.payable_index == idx]
            if not payable_issues:
                corrected_payables.append(payable)
                continue
            corrected = self._correct_one(payable, payable_issues, document_text)
            corrected_payables.append(corrected)

        if not corrected_payables:
            return proposed

        result = FileOutput(file=filename, payables=corrected_payables, declined=proposed.declined)
        return resolve_codes(result, self.master, document_text=document_text)

    def _correct_one(
        self,
        payable: Autodraft,
        issues: list[ValidationIssue],
        document_text: str,
    ) -> Autodraft:
        """Run one corrector inference for a single payable."""
        chain = self.corrector_chain or _corrector_for(
            self.settings.structuring_model, self.settings.structuring_temperature
        )

        from src.validation import MasterDataValidator

        issue_payload = MasterDataValidator.format_issues(issues)
        return chain.invoke(
            {
                "proposed_json": payable.model_dump_json(),
                "issues_json": issue_payload,
                "document_text": document_text,
            }
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _has_errors(issues: list[ValidationIssue]) -> bool:
        return any(i.severity is Severity.ERROR for i in issues)

    @staticmethod
    def _error_count(issues: list[ValidationIssue]) -> int:
        return sum(1 for i in issues if i.severity is Severity.ERROR)


__all__ = ["ValidationLoop", "ValidationResult", "build_corrector_chain"]
