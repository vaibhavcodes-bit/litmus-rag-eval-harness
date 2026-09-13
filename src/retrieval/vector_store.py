from pathlib import Path

from langchain_chroma import Chroma
from src.retrieval.embedder import get_embeddings


CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "litmus_v1"


def get_vector_store(
    persist_directory: Path | str = CHROMA_DIR,
):
    """
    Create or open the Chroma vector store.

    By default, the application uses the production Chroma directory.
    Tests can provide a separate temporary directory.
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )

    return vector_store


def reset_vector_store():
    """
    Delete the existing production Chroma collection
    and create a clean one.

    This is used by the full ingestion process so that
    running ingestion multiple times does not create duplicates.
    """
    vector_store = get_vector_store()

    vector_store.delete_collection()

    return get_vector_store()


def add_documents(
    documents,
    persist_directory: Path | str = CHROMA_DIR,
):
    """
    Add documents to the specified Chroma collection.

    The default directory remains the production Chroma database.
    Tests can provide a temporary directory to avoid contaminating
    production data.
    """
    vector_store = get_vector_store(
        persist_directory=persist_directory,
    )

    if documents:
        vector_store.add_documents(documents)

    return vector_store