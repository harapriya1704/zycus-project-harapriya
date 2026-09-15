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
- **Deterministic ERP guardrails.** Beyond the recompute oracle, the validator hard-blocks the two classic double-counting mistakes that silently inflate a booking: category-header/subtotal rows (`_check_header_rows`) and taxes booked both as a `TAX` line *and* in header `taxes[]` (`_check_tax_duplication`) — both raised as **ERROR** so the corrector must resolve them before converging.
- **Bounded autonomy, zero risk of runaway loops.** The correction loop is hard-capped at `INV_MAX_RETRIES` (default **3**) — the system can improve a record, but it can never loop forever.
- **Every number grounded in the document.** The pipeline never invents a value to "make the total work"; the ERP derives totals from the raw components supplied, so what you see is what was printed.

---

## Technical Architecture

```
                 ┌─────────────────────────────────────────────┐
 documents/*.pdf │                                             │
                 ▼                                             │
      ┌───────────────────┐     ┌─────────────────────────┐   │
      │  READ (PyMuPDF)   │────▶│ EXTRACT (easyocr /    │   │
      │  text layer /     │     │ vision LLM, opt-in)  │   │
      │  image detection  │     │ local OCR by default │   │
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
  - line qty×price vs printed extension and PO references are advisory hints;
  - `_check_header_rows` raises **ERROR** for category-header / subtotal-summary rows (e.g. "PECO ELECTRIC DELIVERY", "TAXES & FEES"): they repeat a section name and carry the section's *aggregate* amount, so keeping them double-counts the ERP's qty×price book;
  - `_check_tax_duplication` raises **ERROR** when the same tax is booked both as a `TAX` line item and in header `taxes[]` — the ERP adds both, inflating gross.
- **Corrector agent** — an LLM handed the proposed JSON *plus* the validation issues *plus* the raw document text; it repairs only flagged fields, never invents values. If a correction inference itself fails (timeout, malformed reply, schema rejection) the failure is contained as `CorrectorFailure`: the flawed **proposal is kept** and the loop stops — the document is never lost and a broken corrector cannot burn the retry budget.
- **The loop** iterates propose → validate → correct until no ERROR-grade issues remain or `max_retries` (default 3) is hit — **guaranteed termination**.

### Multi-provider LLM orchestration

The pipeline is **provider-agnostic by construction**: every LLM — vision transcription *and* text reasoning — is built by `Settings.get_vision_llm()` / `Settings.get_text_llm()`, which resolve the endpoint, credential, and model purely from environment variables. Nothing is hard-coded to a single vendor; the same binary switches between **Hugging Face Serverless Inference** (`https://router.huggingface.co/v1`) and **Groq** (`https://api.groq.com/openai/v1`) by editing `.env`.

| Role | Default (shipped `.env`)            | On Groq                                | On Hugging Face                        |
|------|-------------------------------------|----------------------------------------|----------------------------------------|
| Vision (scan/image pages, opt-in) | `zai-org/GLM-4.5V` on HF | `INV_VISION_BASE_URL` = Groq | `INV_VISION_BASE_URL` = HF router (`INV_VISION_MODEL`) |
| Text (structuring, validation, correction) | `qwen/qwen3.8-27b` on Groq | `INV_TEXT_BASE_URL` = Groq (`INV_TEXT_MODEL`) | `INV_TEXT_BASE_URL` = HF router (`INV_TEXT_MODEL`) |

Key orchestration behaviour:

- **Environment-driven, zero code changes.** `INV_VISION_BASE_URL`, `INV_TEXT_BASE_URL`, `INV_GROQ_BASE_URL`, `INV_VISION_MODEL`, `INV_TEXT_MODEL` — plus `HF_TOKEN` / `GROQ_API_KEY` — fully determine the deployed topology.
- **Hybrid topology (the shipped default).** Vision on the **Hugging Face** router (`zai-org/GLM-4.5V`) and text reasoning on **Groq** (`qwen/qwen3.8-27b`). Vision is **opt-in** (`--use-llm` / `INV_USE_LLM=true`) — a plain `python run.py` never calls it, so the default run spends **zero** vision tokens.
- **Full-HF topology.** Both vision and text reasoning ride the HF Serverless router, independent of Groq's per-model TPD/TPM quotas.
- **Provider validation at startup.** Model availability is probed up front (see *Verification Tools* below) so a misconfigured/un-deployed model is caught before a long batch run begins.

Rationale:

- **Speed.** Open-weight models (Llama-3.3-70B, GPT-OSS-120B, GLM-4.5V) serve full documents in seconds on either LPU (Groq) or Serverless (HF) inference.
- **Cost & sovereignty.** Open weights run cheaply and can be self-hosted; no per-token price lock-in or data-residency constraint inherent to closed APIs.
- **Accuracy where it matters.** Validation is deterministic (master-data cross-reference + ERP oracle), so the *entire* loop converges exactly — the LLM is only the extraction/correction actor, not the source of truth.
- **Portability.** Swapping `INV_VISION_BASE_URL` / `INV_TEXT_BASE_URL` + the model vars re-points the whole pipeline without a single code change.

---

## Production Hardening & Fault Tolerance

Everything below is engineered so a large batch run survives real-world providers
and operators — rate limits, timeouts, partial disk writes, and UI hangs included.

1. **Persistent MD5 transcription cache (`.cache/transcriptions/`).** Every
   rasterised page's transcription is written to disk under a **content-fingerprinted**
   key — an MD5 of the rendered PNG's *bytes* (document + page + image digest), so two
   different pages can never collide even at identical byte lengths. Writes are
   **atomic** (`.tmp` file + `os.replace`), so a hard exit can never leave a
   half-written entry; empty or corrupted cache files are treated as misses and
   re-transcribed (`src/extractor.py`).

2. **Rate-limit & transient-error resilience (`src/llm_retry.py`).** Every provider
   call rides an 8-level cause-chain walk that classifies retryable failures across the
   OpenAI SDK *and* raw HTTPX: 429 (honouring a numeric `Retry-After` header verbatim,
   capped at 300 s), Groq **TPM** 413 `rate_limit_exceeded` (fixed 60 s window wait),
   5xx / 408 / 409 / 425, timeouts, and connection resets — then retries with
   exponential back-off + jitter (capped at 60 s) up to `max_attempts` (default 10).
   A per-document TPM/RPM budget never stalls the batch: one bad file degrades to an
   honest `declined` entry and processing continues.

3. **Zero-token text-layer routing (`src/pipeline.py` & `src/extractor.py`).** Pages
   whose embedded text layer is usable are returned verbatim and **never** contact the
   vision API, and image-only pages are transcribed by **local easyocr CPU OCR** by
   default (`INV_USE_LLM=false` / no `--use-llm`): a plain `python run.py` spends **zero**
   vision tokens. Pass `--use-llm` (or `INV_USE_LLM=true`) to additionally route the
   pages easyocr cannot read to the multimodal model
   (`text_layer_min_chars`, default 80), with consecutive vision calls paced by a
   configurable `INV_VISION_PACING_DELAY` (default 2.0 s) to keep provider TPM/RPM flat
   on multi-page scans. Text-layer re-runs cost **zero** vision tokens.

4. **Thread-safe Streamlit architecture (`app.py`).** The dashboard never blocks on a
   long run: "Run Pipeline" executes on a daemon background thread with a done-flag
   polling loop (the page auto-refreshes, the button is disabled meanwhile). PDF
   rasterisation is memoised with `@st.cache_data` keyed on file content + DPI, errors
   are surfaced in a collapsible expander, and per-PDF results are isolated in session
   state so switching documents never shows stale data.

5. **Fail-fast stall prevention (`src/config.py` + `src/llm_retry.py`).** A single
   LLM call is budgeted by `INV_LLM_ATTEMPTS` (default **1** = fail fast) and any retry
   wait is hard-capped by `INV_LLM_MAX_WAIT` (**60 s**): a provider throttle that would
   force a multi-hundred-second `Retry-After` sleep raises `BenchmarkTimeoutException`
   instead of freezing the process. A large batch therefore degrades an affected file
   to an honest `declined` entry and moves on — the pipeline **never hangs** on a
   congested provider, even when quotas are exhausted.

6. **Production token budgeting (text reasoning).** Text-reasoning calls
   (`src/config.py: text_max_tokens`) request an explicit **8192-token** generation
   window — instead of trusting a serverless endpoint's low default cap — so a complex
   multi-page invoice's full payables JSON (supplier, buyer, payment terms, line items,
   taxes, totals) completes **without mid-payload truncation** even though Groq's
   gpt-oss models expand every line item into its own JSON object (4k–8k output tokens
   on long invoices). A smaller window structurally clipped the JSON and surfaced as
   empty payables (false negative), which is why fast mode shares the same 8192 budget.
   The structured output
   is still enforced through the Pydantic `FileOutput` schema, so a wider window never
   compromises strict schema compliance (malformed replies are rejected, not repaired).
   Oversized transcripts are capped at `INV_MAX_STRUCTURING_CHARS` (default 12000) with
   a **head+tail split** (`_save_truncate`) — the last pages' line items, duty/freight
   charges and grand totals are never dropped from the LLM context.

7. **Automated verification & benchmarking (`scripts/`).**
   - `scripts/check_models.py` — pre-flight health check: probes every candidate
     vision/text model against HF Serverless and Groq, and prints which providers are
     actually deployable on the current account.
   - `scripts/benchmark.py` — comparative performance harness: runs `run.py --validate`
     on representative invoices across the Full-HF / Hybrid / Full-Groq topologies with
     a *fresh* cache each time, then reports wall-clock latency, HTTP 429 /
     rate-limit events, `BenchmarkTimeoutException` aborts, and ERP recompute accuracy
     against the grader's oracle.

---

## Setup Instructions

### 1. Prerequisites

- Python **3.10+** (developed on 3.13).
- A Groq API key, or any OpenAI-compatible endpoint.
- An `HF_TOKEN` (only if you opt into the Vision LLM with `--use-llm`).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure credentials

Copy the template and fill in your key. The shipped hybrid topology needs:

```bash
cp .env.example .env
# edit .env:
#   GROQ_API_KEY=gsk_...                       # text reasoning (structuring, validation, correction)
#   INV_TEXT_BASE_URL=https://api.groq.com/openai/v1
#   INV_TEXT_MODEL=qwen/qwen3.8-27b
#   HF_TOKEN=hf_...                            # vision only — needed just for --use-llm / INV_USE_LLM=true
#   INV_VISION_BASE_URL=https://router.huggingface.co/v1
#   INV_VISION_MODEL=zai-org/GLM-4.5V
#   INV_USE_LLM=false                          # vision is opt-in; a default run is fully local
```

Defaults (current `.env`): text/structuring = **`qwen/qwen3.8-27b`** on **Groq**; vision (opt-in) = **`zai-org/GLM-4.5V`** on the **Hugging Face** router. If your gateway exposes different model ids — and keep the vision model *multimodal* — override per-run or in `.env`:

```bash
export INV_TEXT_MODEL=qwen/qwen3.8-27b
export INV_VISION_MODEL=zai-org/GLM-4.5V
```

### 4. Run the pipeline

```bash
# Full pipeline over documents/ -> output/*.json  (one JSON per PDF)
# Default run is fully local: image pages transcribed by easyocr OCR, the
# Vision LLM is never called (zero vision tokens spent).
python run.py

# Manually trigger the Vision LLM for pages easyocr cannot read
python run.py --use-llm

# With the multi-agent validation loop (day-2)
python run.py --validate

# Fast Testing Mode: bypass vision for text pages, cap retries to 0
python run.py --fast

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
python -m pytest tests -q          # unit/integration tests, no API key required
python -m ruff check src tests scripts run.py erp.py example_check.py check_outputs.py app.py
```

### 6b. Provider health-check & benchmark tools

```bash
# Pre-flight: probe vision/text models on HF Serverless + Groq, print deployable configs
python scripts/check_models.py

# Comparative performance harness: Full-HF / Hybrid / Full-Groq latency,
# rate-limit events, and ERP accuracy (defaults to all architectures)
python scripts/benchmark.py
python scripts/benchmark.py --arch full-hf hybrid
```

### 7. Streamlit dashboard

An interactive UI for reviewing a single invoice end-to-end: rendered PDF pages
beside a live ERP-oracle audit of the extracted output.

```bash
streamlit run app.py
```

Layout:

- **Left panel** — the selected PDF's pages rendered side-by-side via PyMuPDF.
- **Right panel** — a control strip plus three interactive tabs:
  - **ERP Oracle Audit** — feeds the extracted JSON through `erp.erp_book()` and
    compares the printed gross to the recomputed gross, shown as MATCH/MISMATCH
    metric badges.
  - **Extracted JSON** — the structured output formatted per `AUTODRAFT_SCHEMA.md`.
  - **Line Items** — an interactive table of extracted descriptions, quantities,
    unit prices, totals, and tax lines.
- **Sidebar** — pick which PDF from `documents/`, toggle multi-agent validation
  (`--validate`), and press **Run Pipeline** to (re)extract and (re)validate.

If a matching `output/<file>.json` already exists it is loaded for auditing
without re-running the pipeline; clicking **Run Pipeline** regenerates it.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: PDF viewer + ERP oracle audit + JSON/line-item tabs |
| `run.py` | One-command runner (`--validate` enables the agent loop) |
| `check_outputs.py` | ERP bridge: validated outputs → `erp_book` → MATCH/MISMATCH |
| `src/pdf_reader.py` | PDF ingest, text/image detection, rasterisation |
| `src/extractor.py` | easyocr CPU OCR by default + opt-in Vision LLM transcription |
| `src/schemas.py` | Pydantic models mirroring `AUTODRAFT_SCHEMA.md` |
| `src/master_data.py` | O(1)-indexed master-data resolution (no guesses) |
| `src/formatter.py` | LLM structuring chain + provider-agnostic JSON parser |
| `src/validation.py` | Deterministic Validation Agent (master + ERP oracle) |
| `src/agents.py` | Validation loop: proposer/validator/corrector, ≤`max_retries` |
| `src/llm_retry.py` | Retry/back-off wrapper: 429 + TPM + transient-error handling |
| `src/pipeline.py` | Per-file orchestration and `output/` writing |
| `scripts/check_models.py` | Pre-flight provider health-check (HF Serverless + Groq probes) |
| `scripts/benchmark.py` | Comparative latency/429/ERP benchmark across Full-HF / Hybrid / Full-Groq |
| `tests/` | 150+ unit + integration tests (schema, matching, loop, ERP bridge, retries, routing) |
| `erp.py` | The ERP recompute oracle (fixed; graded as-is) |

The original challenge brief is preserved in [`CHALLENGE_BRIEF.md`](CHALLENGE_BRIEF.md).