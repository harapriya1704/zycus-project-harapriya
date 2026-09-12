"""Tests for the end-to-end pipeline (:mod:`src.pipeline`)."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf
from langchain_core.runnables import RunnableLambda
from src.master_data import MasterData
from src.pipeline import process_directory, process_document, write_output
from src.schemas import Autodraft, FileOutput


def _text_pdf(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(40, 40, page.rect.width - 40, page.rect.height - 40), text
    )
    doc.save(path)
    doc.close()
    return path


def _sample_output() -> FileOutput:
    return FileOutput(
        file="X.pdf",
        payables=[
            Autodraft(
                invoice_number="1",
                currency="EUR",
                supplier={"name": "Phocus Direct Communication GmbH", "vat_id": "DE209177122"},
            )
        ],
    )


def test_write_output_shape(tmp_path: Path) -> None:
    result = _sample_output()
    path = write_output(result, tmp_path / "out")
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert set(blob) == {"file", "payables", "declined"}
    assert path.name == "X.json"
    assert blob["payables"][0]["invoice_number"] == "1"


def test_process_document_text_layer_end_to_end(tmp_path: Path) -> None:
    text = "This is an ordinary supplier invoice with enough embedded text to be read by the pipeline without vision. " * 3
    pdf = _text_pdf(tmp_path / "inv.pdf", text)
    master = MasterData.load(Path(__file__).resolve().parent.parent / "master_data")
    fake = RunnableLambda(lambda _: _sample_output())
    result = process_document(pdf, structuring_chain=fake, master=master)
    assert result.file == "inv.pdf"
    assert len(result.payables) == 1
    assert result.payables[0].supplier.supplier_id == "2845695"


def test_process_document_failed_extraction_declines(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pdf"
    result = process_document(missing)
    assert result.payables == []
    assert result.declined[0].doc_type == "ERROR"


def test_process_directory_writes_every_pdf(tmp_path: Path) -> None:
    filler = "Amount of embedded text used by the pipeline test fixture to stay above the image threshold. " * 3
    _text_pdf(tmp_path / "a.pdf", filler + "A")
    _text_pdf(tmp_path / "b.pdf", filler + "B")
    fake = RunnableLambda(lambda _: _sample_output())
    out = tmp_path / "out"
    summary = process_directory(
        tmp_path,
        out,
        structuring_chain=fake,
        master=MasterData.load(Path(__file__).resolve().parent.parent / "master_data"),
    )
    assert summary["files_processed"] == 2
    assert summary["payables"] == 2
    assert (out / "a.json").is_file()
    assert (out / "b.json").is_file()
    assert json.loads((out / "a.json").read_text(encoding="utf-8"))["file"] == "a.pdf"
