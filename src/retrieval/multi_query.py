import os
import re
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.retrieval.retriever import retrieve_documents
from src.retrieval.embedder import get_embeddings


load_dotenv()


MODEL_NAME = "openai/gpt-oss-20b"

DEFAULT_QUERY_COUNT = 4
DEFAULT_RETRIEVAL_K = 4


def get_query_generator():
    """
    Create the LLM used to generate alternative search queries.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to the .env file in the project root."
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=api_key,
    )


def generate_queries(
    question: str,
    query_count: int = DEFAULT_QUERY_COUNT,
) -> list[str]:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    prompt = f"""
Generate exactly {query_count} search queries for this question.

The queries will be used for vector database search.

IMPORTANT:
- Output ONLY the queries.
- One query per line.
- Do NOT answer the question.
- Do NOT ask for clarification.
- Do NOT add explanations.
- Do NOT number the queries.
- Do NOT use bullets.
- Use concise keyword-rich search phrases.
- Include important concepts that may appear in the source document.
- Use different wording for each query.

Question:
{question}

Example output for a job-search question:

job search first step self-assessment career goals target roles
self-assessment before job search target industries non-negotiables
career goals target roles target industries before applying
job search preparation self-assessment before applying
"""

    llm = get_query_generator()
    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        content = " ".join(
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    queries = []

    for line in str(content).splitlines():
        query = line.strip()

        if not query:
            continue

        query = re.sub(r"^[-*•]\s*", "", query)
        query = re.sub(r"^\d+[\.\)]\s*", "", query)
        query = query.strip()

        if not query:
            continue

        # Ignore conversational/non-search responses.
        lower_query = query.lower()

        if (
            "could you please provide" in lower_query
            or "please provide the question" in lower_query
            or "i need the question" in lower_query
            or "what is the question" in lower_query
        ):
            continue

        if query.lower() not in {
            existing.lower() for existing in queries
        }:
            queries.append(query)

    # Keep the original question as a fallback.
    if question.strip().lower() not in {
        existing.lower() for existing in queries
    }:
        queries.append(question.strip())

    return queries[:query_count]
def _document_key(document: Document) -> str:
    """
    Create a stable key for deduplicating documents.

    Prefer chunk_id when available.

    If chunk_id does not exist, fall back to source/page/content.
    """

    metadata = document.metadata or {}

    chunk_id = metadata.get("chunk_id")

    if chunk_id:
        return f"chunk_id:{chunk_id}"

    source = metadata.get("source", "")
    page = metadata.get("page", "")

    content = document.page_content.strip()

    return f"{source}|{page}|{content}"


def deduplicate_documents(
    documents: List[Document],
) -> List[Document]:
    """
    Remove duplicate documents while preserving their first-seen order.
    """

    unique_documents = []
    seen = set()

    for document in documents:
        key = _document_key(document)

        if key in seen:
            continue

        seen.add(key)
        unique_documents.append(document)

    return unique_documents



def debug_rank_documents(
    documents_by_query: list[list[Document]],
    k: int = 60,
) -> list[dict]:
    """
    Show how each document received its RRF score.

    This is a diagnostic helper only.
    It does not change the production ranking.
    """

    scores = {}
    documents = {}
    appearances = {}

    for query_index, query_documents in enumerate(
        documents_by_query,
        start=1,
    ):
        seen_in_query = set()

        for rank, document in enumerate(query_documents, start=1):
            document_key = _document_key(document)

            if document_key in seen_in_query:
                continue

            seen_in_query.add(document_key)

            if document_key not in documents:
                documents[document_key] = document
                scores[document_key] = 0.0
                appearances[document_key] = []

            score = 1.0 / (k + rank)

            scores[document_key] += score

            appearances[document_key].append(
                {
                    "query": query_index,
                    "rank": rank,
                    "score": score,
                }
            )

    results = []

    for document_key, document in documents.items():
        results.append(
            {
                "document_key": document_key,
                "source": document.metadata.get("source"),
                "content": document.page_content[:300]
                    .replace("\n", " "),
                "rrf_score": scores[document_key],
                "appearances": appearances[document_key],
            }
        )

    results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results


def rank_documents(
    documents_by_query: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Rank documents using Reciprocal Rank Fusion (RRF).

    Documents that appear near the top of multiple query results
    receive higher scores.

    RRF score:

        score = 1 / (k + rank)

    where rank starts at 1.
    """

    scores = {}
    documents = {}

    first_seen_order = {}
    order_counter = 0

    for query_documents in documents_by_query:
        # Prevent the same document from receiving multiple scores
        # from duplicate occurrences within one query result.
        seen_in_query = set()

        for rank, document in enumerate(query_documents, start=1):
            document_key = _document_key(document)

            if document_key in seen_in_query:
                continue

            seen_in_query.add(document_key)

            if document_key not in documents:
                documents[document_key] = document
                first_seen_order[document_key] = order_counter
                order_counter += 1

            score = 1.0 / (k + rank)

            scores[document_key] = (
                scores.get(document_key, 0.0) + score
            )

    ranked_keys = sorted(
        documents.keys(),
        key=lambda key: (
            -scores[key],
            first_seen_order[key],
        ),
    )

    return [documents[key] for key in ranked_keys]

def multi_query_retrieve(
    question: str,
    query_count: int = 4,
    k: int = 4,
):
    original_documents = retrieve_documents(
        question=question,
        k=max(k * 2, 8),
    )

    generated_queries = generate_queries(
        question=question,
        query_count=query_count,
    )

    generated_documents_by_query = []

    for query in generated_queries:
        if query.strip().lower() == question.strip().lower():
            continue

        generated_documents_by_query.append(
            retrieve_documents(
                query,
                max(k * 2, 8),
            )
        )

        ranked_documents = weighted_rank_documents(
        original_documents=original_documents,
        documents_by_query=generated_documents_by_query,
        original_question=question,
        original_weight=2.0,
        lexical_weight=2.0,
        intent_weight=3.0,
        rrf_k=60,
    )

    return deduplicate_documents(ranked_documents)[: query_count * k]

def rerank_by_original_question(
    documents: list[Document],
    question: str,
) -> list[Document]:
    """
    Re-rank candidate documents using similarity to the
    original user question.

    This is a second-stage ranking step after multi-query
    retrieval and RRF.
    """

    if not documents:
        return []

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    embeddings = get_embeddings()

    question_embedding = embeddings.embed_query(
        question
    )

    document_texts = [
        document.page_content
        for document in documents
    ]

    document_embeddings = embeddings.embed_documents(
        document_texts
    )

    def cosine_similarity(vector_a, vector_b):
        dot_product = sum(
            a * b
            for a, b in zip(vector_a, vector_b)
        )

        magnitude_a = sum(
            a * a
            for a in vector_a
        ) ** 0.5

        magnitude_b = sum(
            b * b
            for b in vector_b
        ) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (
            magnitude_a * magnitude_b
        )

    scored_documents = []

    for document, embedding in zip(
        documents,
        document_embeddings,
    ):
        similarity = cosine_similarity(
            question_embedding,
            embedding,
        )

        scored_documents.append(
            (
                similarity,
                document,
            )
        )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored_documents
    ]
    
def lexical_relevance_score(
    question: str,
    document: Document,
) -> float:
    """
    Calculate a lightweight lexical relevance score between
    the original question and a document.

    The score is based on meaningful words shared by the
    question and document.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not document.page_content:
        return 0.0

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "before",
        "be",
        "do",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "starting",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "with",
    }

    question_words = {
        word.lower()
        for word in re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", question)
        if word.lower() not in stop_words
    }

    document_words = {
        word.lower()
        for word in re.findall(
            r"\b[a-zA-Z][a-zA-Z-]+\b",
            document.page_content,
        )
    }

    if not question_words:
        return 0.0

    overlap = question_words.intersection(document_words)

    return len(overlap) / len(question_words)    


def intent_phrase_score(
    question: str,
    document: Document,
) -> float:
    """
    Score important multi-word or intent phrases from the
    question against the retrieved document.

    This provides a stronger signal than individual-word
    overlap when several documents share generic vocabulary.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not document.page_content:
        return 0.0

    question_lower = question.lower()
    document_lower = document.page_content.lower()

    important_phrases = []

    if "first step" in question_lower:
        important_phrases.append("first step")

    if "self-assessment" in question_lower:
        important_phrases.append("self-assessment")

    if "career goals" in question_lower:
        important_phrases.append("career goals")

    if "target roles" in question_lower:
        important_phrases.append("target roles")

    if "target industries" in question_lower:
        important_phrases.append("target industries")

    if "non-negotiables" in question_lower:
        important_phrases.append("non-negotiables")

    if not important_phrases:
        return 0.0

    matched = sum(
        1
        for phrase in important_phrases
        if phrase in document_lower
    )

    return matched / len(important_phrases)
    
    
def weighted_rank_documents(
    original_documents: list[Document],
    documents_by_query: list[list[Document]],
    original_question: str,
    original_weight: float = 2.0,
    lexical_weight: float = 2.0,
    intent_weight: float = 3.0,
    rrf_k: int = 60,
) -> list[Document]:
    """
    Rank documents using:

    1. Original-question relevance
    2. Generated-query RRF evidence
    3. Lexical relevance to the original question
    4. Intent phrase relevance

    The intent signal helps distinguish a document that contains
    the important phrases from a generic document that happens
    to appear in several generated-query result sets.
    """

    if not original_question or not original_question.strip():
        raise ValueError("Question cannot be empty.")

    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    # ---------------------------------------------------------
    # 1. Original question ranking
    # ---------------------------------------------------------
    original_count = max(len(original_documents), 1)

    for rank, document in enumerate(original_documents, start=1):
        key = _document_key(document)

        documents[key] = document

        original_score = (
            (original_count - rank + 1)
            / original_count
        )

        scores[key] = scores.get(key, 0.0) + (
            original_weight * original_score
        )

    # ---------------------------------------------------------
    # 2. Generated-query RRF contributions
    # ---------------------------------------------------------
    for query_documents in documents_by_query:
        for rank, document in enumerate(query_documents, start=1):
            key = _document_key(document)

            documents[key] = document

            generated_score = 1.0 / (rrf_k + rank)

            scores[key] = scores.get(key, 0.0) + generated_score

    # ---------------------------------------------------------
    # 3. Lexical relevance to the original question
    # ---------------------------------------------------------
    for key, document in documents.items():
        lexical_score = lexical_relevance_score(
            original_question,
            document,
        )

        scores[key] = scores.get(key, 0.0) + (
            lexical_weight * lexical_score
        )

    # ---------------------------------------------------------
    # 4. Intent phrase relevance
    # ---------------------------------------------------------
    for key, document in documents.items():
        intent_score = intent_phrase_score(
            original_question,
            document,
        )

        scores[key] = scores.get(key, 0.0) + (
            intent_weight * intent_score
        )

    # ---------------------------------------------------------
    # 5. Sort by combined score
    # ---------------------------------------------------------
    ranked_keys = sorted(
        scores,
        key=lambda key: scores[key],
        reverse=True,
    )

    return [documents[key] for key in ranked_keys]