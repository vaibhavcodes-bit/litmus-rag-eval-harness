from src.retrieval.retriever import retrieve_documents
from src.retrieval.decomposition import decompose_and_retrieve
from src.generation.llm_client import generate_answer


def _documents_to_sources(documents):
    """
    Convert LangChain Documents into the source format
    returned by the API.
    """

    sources = []

    for document in documents:
        sources.append(
            {
                "source": document.metadata.get("source"),
                "page": document.metadata.get("page"),
                "content": document.page_content,
            }
        )

    return sources


def answer_question(
    question: str,
    k: int = 4,
    mode: str = "v1",
):
    """
    Run the RAG pipeline.

    Supported modes:

        v1:
            Question
                ↓
            Baseline Retrieval
                ↓
            Context
                ↓
            LLM

        v4:
            Question
                ↓
            Decompose
                ↓
            Sub-question Retrieval
                ↓
            Combined Context
                ↓
            LLM
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if mode not in {"v1", "v4"}:
        raise ValueError(
            "mode must be either 'v1' or 'v4'."
        )

    # ---------------------------------------------------------
    # V1 — Baseline RAG
    # ---------------------------------------------------------
    if mode == "v1":

        documents = retrieve_documents(
            question=question,
            k=k,
        )

        if not documents:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
            }

        context_parts = []

        for document in documents:
            context_parts.append(
                document.page_content
            )

        context = "\n\n".join(context_parts)

        answer = generate_answer(
            question=question,
            context=context,
        )

        sources = _documents_to_sources(
            documents
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    # ---------------------------------------------------------
    # V4 — Query Decomposition
    # ---------------------------------------------------------
    if mode == "v4":

        result = decompose_and_retrieve(
            question=question,
            k=k,
        )

        sub_questions = result["sub_questions"]

        documents_by_sub_question = result[
            "documents_by_sub_question"
        ]

        context = result["context"]

        # Flatten documents so the API keeps a simple
        # sources[] response structure.
        all_documents = []

        for documents in documents_by_sub_question:
            all_documents.extend(documents)

        if not all_documents:
            return {
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "sub_questions": sub_questions,
            }

        answer = generate_answer(
            question=question,
            context=context,
        )

        sources = _documents_to_sources(
            all_documents
        )

        return {
            "answer": answer,
            "sources": sources,
            "sub_questions": sub_questions,
        }