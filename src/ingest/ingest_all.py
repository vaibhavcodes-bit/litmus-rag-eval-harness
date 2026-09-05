from pathlib import Path

from src.ingest.loader import load_pdf
from src.ingest.chunker import chunk_documents
from src.retrieval.vector_store import add_documents


RAW_DATA_DIR = Path("data/raw")


def ingest_all_pdfs():
    """
    Load every PDF from data/raw, split it into chunks,
    and add the chunks to the persistent Chroma vector store.
    """

    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            "No PDF files found in data/raw/"
        )

    total_documents = 0
    total_chunks = 0

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        documents = load_pdf(str(pdf_file))
        chunks = chunk_documents(documents)

        add_documents(chunks)

        total_documents += len(documents)
        total_chunks += len(chunks)

        print(f"  Pages loaded : {len(documents)}")
        print(f"  Chunks created: {len(chunks)}")

    print("\n" + "=" * 50)
    print("Ingestion complete")
    print("=" * 50)
    print(f"PDF files      : {len(pdf_files)}")
    print(f"Pages loaded   : {total_documents}")
    print(f"Chunks stored  : {total_chunks}")


if __name__ == "__main__":
    ingest_all_pdfs()