from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parents[1]

V4_TEST_DIR = BASE_DIR / "data" / "v4_test"
V4_CHROMA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "v4_benchmark_chroma"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_v4_documents() -> List[Document]:
    """Load all documents from the isolated V4 benchmark corpus."""

    documents = []

    for file_path in sorted(V4_TEST_DIR.glob("*.txt")):
        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_path": str(file_path),
                },
            )
        )

    if not documents:
        raise RuntimeError(
            f"No V4 benchmark documents found in {V4_TEST_DIR}"
        )

    return documents


def chunk_v4_documents(
    documents: List[Document],
) -> List[Document]:
    """Split V4 benchmark documents into retrieval chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['source']}::chunk_{index}"
        )

    return chunks


def get_embeddings():
    """Create the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def build_v4_vector_store():
    """Build the isolated Chroma store for the V4 benchmark."""

    documents = load_v4_documents()

    chunks = chunk_v4_documents(documents)

    V4_CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store = Chroma(
        collection_name="litmus_v4_benchmark",
        embedding_function=get_embeddings(),
        persist_directory=str(V4_CHROMA_DIR),
    )

    existing = vector_store.get()

    if existing and existing.get("ids"):
        vector_store.delete(
            ids=existing["ids"]
        )

    vector_store.add_documents(
        chunks,
        ids=[
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ],
    )

    return vector_store


def retrieve_v4_benchmark(
    vector_store,
    question: str,
    k: int = 4,
) -> List[Document]:
    """Retrieve top-k documents for a benchmark question."""

    if not question or not question.strip():
        raise ValueError(
            "question cannot be empty"
        )

    return vector_store.similarity_search(
        question,
        k=k,
    )


if __name__ == "__main__":
    store = build_v4_vector_store()

    documents = retrieve_v4_benchmark(
        store,
        "What are the benefits of remote work and on-site work?",
        k=4,
    )

    print(
        f"Retrieved documents: {len(documents)}"
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"\nRank {index}: "
            f"{document.metadata.get('source')}"
        )

        print(
            document.page_content[:300]
        )