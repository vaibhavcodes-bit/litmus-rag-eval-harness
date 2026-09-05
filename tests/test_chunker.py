from pathlib import Path

from src.ingest.loader import load_pdf
from src.ingest.chunker import chunk_documents


def test_chunk_documents():
    pdf_files = list(Path("data/raw").glob("*.pdf"))

    assert pdf_files, (
        "No PDF found. Put at least one PDF inside data/raw/"
    )

    documents = load_pdf(str(pdf_files[0]))

    chunks = chunk_documents(documents)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.page_content.strip() != ""
        assert len(chunk.page_content) <= 500

        assert "source" in chunk.metadata
        assert "page" in chunk.metadata