from pathlib import Path

from langchain_chroma import Chroma

from src.retrieval.embedder import get_embeddings


CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "litmus_v1"


def get_vector_store():
    """
    Create or load the persistent Chroma vector store.
    """

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    return vector_store


def add_documents(documents):
    """
    Add document chunks to the persistent vector store.
    """

    vector_store = get_vector_store()

    if documents:
        vector_store.add_documents(documents)

    return vector_store