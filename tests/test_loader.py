from pathlib import Path

from src.ingest.loader import load_pdf


def test_load_pdf():
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    assert pdf_files, (
        "No PDF found. Put at least one PDF inside data/raw/"
    )

    documents = load_pdf(str(pdf_files[0]))

    assert len(documents) > 0

    first_document = documents[0]

    assert first_document.page_content.strip() != ""

    assert "source" in first_document.metadata

    assert "page" in first_document.metadata