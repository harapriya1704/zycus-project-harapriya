"""Streamlit dashboard for the Agentic ERP Automation Pipeline.

    streamlit run app.py

Layout:
- Left panel  — rendered PDF pages side-by-side (PyMuPDF).
- Right panel — a control strip (sidebar file selector + "Run Pipeline") and
  three interactive tabs:

    * ERP Oracle Audit — feeds the extracted JSON through ``erp.erp_book`` and
      compares the printed gross to the recomputed gross with MATCH/MISMATCH
      metric badges.
    * Extracted JSON    — the structured output formatted per AUTODRAFT_SCHEMA.md.
    * Line Items        — an interactive table of the extracted line items
      (descriptions, quantities, unit prices, totals, tax lines).
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Agentic ERP Pipeline", layout="wide")


def _documents_dir() -> Path:
    from src.config import get_settings

    return get_settings().resolved_documents_dir()


@st.cache_data(show_spinner="Rendering PDF pages...")
def _page_pngs(pdf_path: str, mtime_ns: int, dpi: int) -> list[tuple[int, bytes]]:
    """Rasterise every page of *pdf_path* to PNG bytes (cached by mtime).

    Rendering happens once per (file content, dpi); switching PDFs or re-runs
    of the app do not re-rasterise the same file.
    """
    import pymupdf

    pages: list[tuple[int, bytes]] = []
    with pymupdf.open(pdf_path) as doc:
        for page_no in range(len(doc)):
            pix = doc.load_page(page_no).get_pixmap(dpi=dpi)
            pages.append((page_no + 1, pix.tobytes("png")))
    return pages


def _render_pdf_pages(pdf_path: Path, dpi: int = 110) -> None:
    """Side-by-side the PDF's pages in the left panel, shrunk to the column.

    Every page image is set to ``width="stretch"`` (the Streamlit ≥1.50 binding
    for full-container width) so it can never exceed the left column and spill
    into the right-hand results panel.
    """
    mtime_ns = int(pdf_path.stat().st_mtime_ns)
    pages = _page_pngs(str(pdf_path), mtime_ns, dpi)
    for start in range(0, len(pages), 2):
        row = st.columns(2, gap="small")
        for offset in range(2):
            slot = start + offset
            if slot >= len(pages):
                row[offset].caption("&nbsp;")
                continue
            page_no, png = pages[slot]
            row[offset].image(png, caption=f"Page {page_no}", width="stretch")


def _load_result(pdf_path: Path) -> dict | None:
    """Return the stored output JSON for *pdf_path*, or ``None``."""
    out = _resolve_output_dir() / f"{pdf_path.stem}.json"
    if not out.is_file():
        return None
    try:
        return json.loads(out.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt/partial output file must not crash the whole dashboard.
        return None


def _resolve_output_dir() -> Path:
    from src.config import get_settings

    return get_settings().resolved_output_dir()


def _run_pipeline(pdf_path: Path, validate: bool) -> dict:
    """Run the extract-structure-(validate) pipeline for one PDF and persist it."""
    from src.agents import ValidationLoop
    from src.config import get_settings
    from src.master_data import MasterData
    from src.pipeline import process_document, write_output

    cfg = get_settings()
    master = MasterData.load_default(settings=cfg)
    loop = ValidationLoop(master=master, settings=cfg) if validate else None
    result = process_document(pdf_path, settings=cfg, master=master, validation_loop=loop)
    write_output(result, cfg.resolved_output_dir())
    return json.loads(result.to_json())


def _badge(match: bool) -> str:
    icon, colour = ("✓", "#1a8f4e") if match else ("✕", "#c24a4a")
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 12px;'
        f'border-radius:12px;font-weight:600">{icon} {"MATCH" if match else "MISMATCH"}</span>'
    )


def _audit_payable(payable: dict) -> None:
    from erp import erp_book
    from src.schemas import Autodraft

    try:
        draft = Autodraft.model_validate(payable)
        booked = erp_book(draft.model_dump(mode="json"))
        printed = float(str(payable.get("gross_total") or "0") or 0)
        delta = abs(booked["will_book_gross"] - printed)
        match = delta < 0.01
    except Exception as exc:  # noqa: BLE001 - one malformed payable must not crash the tab
        st.error(f"ERP audit failed for this payable: {type(exc).__name__}: {exc}")
        return

    st.markdown(_badge(match), unsafe_allow_html=True)
    left, right = st.columns(2)
    currency = str(booked.get("currency") or "").strip()
    left.metric("Printed gross", f'{currency}{printed:,.2f}'.strip())
    right.metric(
        "Recomputed gross (erp_book)",
        f'{currency}{booked["will_book_gross"]:,.2f}'.strip(),
        delta=f"{booked['will_book_gross'] - printed:,.2f}",
    )


def _launch_pipeline(pdf_path: Path, validate: bool) -> None:
    """Run the pipeline for one PDF off the UI thread.

    The browser stays responsive while Groq rate-limit back-off retries; the
    result (or a clean error) lands back in session state for the next rerun.
    """

    def _work() -> None:
        try:
            st.session_state["pipeline_result"] = _run_pipeline(pdf_path, validate)
            st.session_state["pipeline_error"] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never kill the thread
            from src.llm_retry import describe_error

            st.session_state["pipeline_result"] = None
            st.session_state["pipeline_error"] = describe_error(exc)
        finally:
            st.session_state["pipeline_done"] = True

    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_done"] = False
    st.session_state["pipeline_result"] = None
    st.session_state["pipeline_error"] = None
    st.session_state["pipeline_thread"] = threading.Thread(target=_work, daemon=True)
    st.session_state["pipeline_thread"].start()


def _render_audit_tab(payload: dict | None) -> None:
    if payload is None:
        st.info("No extracted output yet — run the pipeline from the sidebar.")
        return
    if not payload.get("payables"):
        st.warning("No bookable payables in this result.")
        for entry in payload.get("declined", []):
            st.write(f"Declined ({entry.get('doc_type', 'declined')}): {entry.get('reason', '')}")
        return
    for i, payable in enumerate(payload["payables"]):
        st.subheader(f"Payable {i + 1} — {payable.get('invoice_number') or 'no number'}")
        _audit_payable(payable)
        st.divider()


def _render_json_tab(payload: dict | None) -> None:
    if payload is None:
        st.info("No extracted output yet — run the pipeline from the sidebar.")
        return
    st.caption("Formatted per AUTODRAFT_SCHEMA.md — raw components only; the ERP derives the totals.")
    st.json(payload, expanded=True)


def _render_line_items_tab(payload: dict | None) -> None:
    if payload is None:
        st.info("No extracted output yet — run the pipeline from the sidebar.")
        return
    for i, payable in enumerate(payload.get("payables", [])):
        st.subheader(f"Payable {i + 1} — {payable.get('invoice_number') or 'no number'}")
        lines = payable.get("line_items", [])
        if not lines:
            st.caption("No line items extracted.")
            continue
        rows = [
            {
                "Description": li.get("description", ""),
                "Type": li.get("item_type", ""),
                "Qty": li.get("quantity", ""),
                "Unit price": li.get("unit_price", ""),
                "Total": li.get("total", ""),
                "Discount": li.get("discount", ""),
                "Tax amount": li.get("tax_amount", ""),
                "UoM": li.get("uom", ""),
            }
            for li in lines
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        if payable.get("taxes"):
            st.write("**Header taxes**")
            st.table(
                [
                    {
                        "Name": t.get("tax_name", ""),
                        "Rate %": t.get("tax_rate", ""),
                        "Amount": t.get("tax_amount", ""),
                        "Code": t.get("tax_type_code", ""),
                    }
                    for t in payable["taxes"]
                ]
            )


def _main() -> None:
    st.title("Agentic ERP Automation Pipeline")
    st.caption("Turn supplier PDFs into bookable ERP autodrafts — validated against the ERP oracle.")

    pdfs = sorted(_documents_dir().glob("*.pdf"))
    if not pdfs:
        st.warning(f"No PDFs found under {_documents_dir()} — drop invoices into `documents/`.")
        return

    st.session_state.setdefault("results", {})
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pipeline_done", False)
    st.session_state.setdefault("pipeline_result", None)
    st.session_state.setdefault("pipeline_error", None)
    st.session_state.setdefault("last_error", None)

    with st.sidebar:
        st.header("Invoice selector")
        names = [p.name for p in pdfs]
        selected = st.selectbox("PDF file", names, index=0)
        pdf_path = _documents_dir() / selected
        validate = st.checkbox("Multi-agent validation (--validate)", value=True)
        running = st.session_state["pipeline_running"]

        if running:
            st.button("Run Pipeline", disabled=True, type="primary")
            st.info("Pipeline running in the background — retries on rate limits are automatic.")
        elif st.button("Run Pipeline", type="primary"):
            st.session_state["results"].pop(pdf_path.stem, None)  # never show stale data
            st.session_state["last_error"] = None
            _launch_pipeline(pdf_path, validate)

    if st.session_state["pipeline_running"]:
        if st.session_state["pipeline_done"]:
            new_result = st.session_state["pipeline_result"]
            error = st.session_state["pipeline_error"]
            if new_result is not None:
                st.session_state["results"][pdf_path.stem] = new_result
            st.session_state["pipeline_running"] = False
            st.session_state["pipeline_done"] = False
            st.session_state["pipeline_result"] = None
            st.session_state["pipeline_error"] = None
            if error is not None:
                # Kept for the expander in the results column (shown once).
                st.session_state["last_error"] = error
        else:
            # Busy-wait rerun: the UI thread must keep returning so the page
            # stays interactive; each rerun re-checks the done flag.
            time.sleep(0.25)
            st.info("Pipeline running — this page refreshes automatically when it completes.")
            st.rerun()

    if pdf_path.stem not in st.session_state["results"]:
        stored = _load_result(pdf_path)
        if stored is not None:
            st.session_state["results"][pdf_path.stem] = stored

    result = st.session_state["results"].get(pdf_path.stem)

    col_pdf, col_results = st.columns([5, 7], gap="large")
    with col_pdf:
        st.subheader("Source document")
        with st.container():
            _render_pdf_pages(pdf_path)

    with col_results:
        st.subheader("Pipeline output")
        if st.session_state.get("last_error"):
            # Long API/rate-limit error logs live in a collapsible expander so
            # they can never stretch the page layout or cover the audit tabs.
            with st.expander("⚠️ Execution Details / Rate Limit Logs"):
                st.error(st.session_state["last_error"])
        tab_audit, tab_json, tab_lines = st.tabs(
            ["ERP Oracle Audit", "Extracted JSON", "Line Items"]
        )
        with tab_audit:
            _render_audit_tab(result)
        with tab_json:
            _render_json_tab(result)
        with tab_lines:
            _render_line_items_tab(result)


if __name__ == "__main__":
    _main()
