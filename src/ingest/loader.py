from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a PDF and return LangChain Document objects.

    Each document represents one PDF page.
    The source filename is stored in metadata.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {pdf_path}")

    loader = PyPDFLoader(str(path))

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = path.name

    return documents