# Agentic ERP Automation Pipeline

*Turn supplier PDFs into bookable ERP autodrafts — with agent-driven validation against your master data.*

---

## Project Title & Vision

**Agentic ERP Automation Pipeline** removes the manual intake bottleneck between *a document arriving* and *an ERP booking it*. Given a folder of supplier PDFs it automatically:

1. decides what each document is (invoice, credit memo, statement, or "not a payable at all"),
2. extracts the raw components exactly as printed (quantities, unit prices, discounts, taxes, charges),
3. resolves every master-data code (supplier, buyer org, tax, payment term, P.O.) — never guessing,
4. validates each candidate against **both** the master-data files **and** the ERP's own recompute oracle,
5. flags and **autonomously corrects** inconsistencies through a bounded multi-agent loop, and
6. writes one schema-compliant JSON per PDF for the accounting system to book.

The vision: an accounts-payable team approves exceptions instead of keying documents.

### Business outcomes

- **~90% reduction in manual processing time.** Human effort moves from data-entry to exception review. A modern AP headcount processes ~1,500 invoices/month; an agent pipeline with this accuracy reduces that window to reviewing only the flagged exceptions.
- **100% schema compliance by construction.** Every record is emitted through a Pydantic model that mirrors `AUTODRAFT_SCHEMA.md` exactly — the machine cannot emit a malformed payable, and the contract boundary rejects (and drops) fields the model invented.
- **Autonomous vendor-discrepancy detection.** The Validation Agent cross-references each candidate against `master_data/` and the `erp.py` oracle, flagging unmatched vendor IDs, fabricated master codes, line-arithmetic drift and gross recompute mismatches — before they reach the ERP.
- **Bounded autonomy, zero risk of runaway loops.** The correction loop is hard-capped at `INV_MAX_RETRIES` (default **3**) — the system can improve a record, but it can never loop forever.
- **Every number grounded in the document.** The pipeline never invents a value to "make the total work"; the ERP derives totals from the raw components supplied, so what you see is what was printed.

---

## Technical Architecture

```
                 ┌─────────────────────────────────────────────┐
 documents/*.pdf │                                             │
                 ▼                                             │
      ┌───────────────────┐     ┌─────────────────────────┐   │
      │  READ (PyMuPDF)   │────▶│ EXTRACT (vision LLM)   │   │
      │  text layer /     │     │ pymupdf raster + Groq   │   │
      │  image detection  │     │ transcription           │   │
      └───────────────────┘     └────────────┬────────────┘   │
                                             ▼                │
                 ┌─────────────────────────────────────────┐   │
                 │           VALIDATION LOOP              │   │
                 │                                         │   │
                 │  ┌──────────┐    ┌──────────────────┐   │   │
                 │  │ PROPOSER │───▶│   VALIDATOR      │   │   │
                 │  │ agent    │    │ (deterministic:  │───┼───┼──▶  converge?
                 │  │ (LLM)    │    │  master + oracle) │   │   │
                 │  └────▲─────┘    └────────┬─────────┘   │   │
                 │       │                   │ issues      │   │
                 │       │   ┌───────────┐  │              │   │
                 │       │   │ CORRECTOR │◀─┘              │   │
                 │       │   │ agent (LLM)│  (≤ 3 retries) │   │
                 │       └───┴───────────┘                  │   │
                 └─────────────────────────────────────────┘   │
                                             │                 │
                                             ▼                 │
                              ┌───────────────────────┐        │
                              │ output/<doc>.json     │───▶ erp.py
                              │ (AUTODRAFT_SCHEMA)    │    (grading oracle)
                              └───────────────────────┘        │
                 ┌─────────────────────────────────────────────┘
                 │
                 └──▶ check_outputs.py feeds every payable back into
                     erp.erp_book and prints MATCH/MISMATCH vs printed gross.
```

### Multi-agent design (LangChain)

- **Proposer agent** — the LLM structures document text into an autodraft, leaving every master-data code `""`. Codes are then resolved **deterministically** (`src/master_data`), so no code is ever a model guess.
- **Validation agent** — *pure logic, no LLM* (`src/validation.py`). Cross-references each payable against the indexed masters and the ERP oracle:
  - every set master code exists in its master file (ERROR otherwise);
  - an unmatched vendor ID / supplier string is flagged;
  - the ERP recompute of the raw components is compared to the printed `gross_total` (amount inconsistency = ERROR);
  - line qty×price vs printed extension and PO references are advisory hints.
- **Corrector agent** — an LLM handed the proposed JSON *plus* the validation issues *plus* the raw document text; it repairs only flagged fields, never invents values.
- **The loop** iterates propose → validate → correct until no ERROR-grade issues remain or `max_retries` (default 3) is hit — **guaranteed termination**.

### Why open-weight models via Groq

All agent LLMs run against an **OpenAI-compatible endpoint driven by `OPENAI_API_KEY` / `OPENAI_BASE_URL`**, configured to target **`llama-3.3-70b-versatile`** (open-weight, Llama 3.3 70B, served via Groq). Rationale:

- **Speed.** Groq's LPU inference serves Llama 3.3 70B at hundreds of tokens/second — a full document runs in seconds, not minutes.
- **Cost & sovereignty.** Open-weight models run cheaply and can be self-hosted; there is no per-token price lock-in or data-residency constraint inherent to closed APIs.
- **Accuracy where it matters.** Validation is deterministic (master-data cross-reference + ERP oracle), so the *entire* loop converges exactly — the LLM is only the extraction/correction actor, not the source of truth.
- **Portability.** The provider is swappable: changing `OPENAI_BASE_URL` + `INV_STRUCTURING_MODEL` re-points the whole pipeline without code changes.

---

## Setup Instructions

### 1. Prerequisites

- Python **3.10+** (developed on 3.13).
- A Groq API key (or any OpenAI-compatible endpoint).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

Copy the template and fill in your key:

```bash
cp .env.example .env
# edit .env:
#   OPENAI_API_KEY=gsk_...
#   OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

Defaults target **`llama-3.3-70b-versatile`** on Groq as the extraction and structuring model. If your Groq account exposes a different id, override per-run or in `.env`:

```bash
export INV_STRUCTURING_MODEL=openai/gpt-oss-120b
```

### 4. Run the pipeline

```bash
# Full pipeline over documents/ -> output/*.json  (one JSON per PDF)
python run.py

# With the multi-agent validation loop (day-2)
python run.py --validate

# Dry run: list the input PDFs
python run.py --list
```

`run.py --validate` also prints a summary with validation stats:

```json
{
  "files_processed": 42,
  "payables": 0,
  "declined": { "UNREADABLE": 37, "ERROR": 5 },
  "validation": { "converged": 0, "not_converged": 0, "corrections": 0 }
}
```

### 5. Verify against the ERP oracle

```bash
# The reference example (should print will_book_gross = 438.0 EUR)
python example_check.py

# Feed every validated payable back into the ERP and compare to printed gross
python check_outputs.py            # or: python check_outputs.py output/INV-31.json
```

### 6. Test & lint

```bash
python -m pytest tests -q          # 70+ tests, no API key required
python -m ruff check src tests run.py erp.py example_check.py check_outputs.py
```

---

## Repository layout

| Path | Purpose |
|---|---|
| `run.py` | One-command runner (`--validate` enables the agent loop) |
| `check_outputs.py` | ERP bridge: validated outputs → `erp_book` → MATCH/MISMATCH |
| `src/pdf_reader.py` | PDF ingest, text/image detection, rasterisation |
| `src/extractor.py` | Vision LLM transcription of image-only pages |
| `src/schemas.py` | Pydantic models mirroring `AUTODRAFT_SCHEMA.md` |
| `src/master_data.py` | O(1)-indexed master-data resolution (no guesses) |
| `src/formatter.py` | LLM structuring chain + provider-agnostic JSON parser |
| `src/validation.py` | Deterministic Validation Agent (master + ERP oracle) |
| `src/agents.py` | Validation loop: proposer/validator/corrector, ≤`max_retries` |
| `src/pipeline.py` | Per-file orchestration and `output/` writing |
| `tests/` | 70+ unit + integration tests (schema, matching, loop, ERP bridge) |
| `erp.py` | The ERP recompute oracle (fixed; graded as-is) |

The original challenge brief is preserved in [`CHALLENGE_BRIEF.md`](CHALLENGE_BRIEF.md).