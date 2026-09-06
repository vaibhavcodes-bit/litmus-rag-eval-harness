from pathlib import Path
import sys


# ----------------------------------------------------------------------
# Make project root importable
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.multi_query import (
    _document_key,
    generate_queries,
    lexical_relevance_score,
    retrieve_documents,
    weighted_rank_documents,
)


QUESTION = "What is the recommended first step before starting a job search?"

QUERY_COUNT = 4
RETRIEVAL_K = 4


def print_document(
    rank: int,
    document,
    score: float | None = None,
):
    print("=" * 70)
    print(f"RANK {rank}")
    print("=" * 70)

    source = document.metadata.get("source", "unknown")
    chunk_id = document.metadata.get("chunk_id")

    print(f"Source: {source}")
    print(f"Chunk ID: {chunk_id}")

    if score is not None:
        print(f"Combined Score: {score:.6f}")

    print()
    print("Content:")
    print(document.page_content[:1200])
    print()


def main():
    print("=" * 70)
    print("V3 WEIGHTED + LEXICAL RANKING DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Original question:")
    print(QUESTION)

    # ------------------------------------------------------------------
    # 1. Generate paraphrased queries
    # ------------------------------------------------------------------

    print()
    print("-" * 70)
    print("GENERATED QUERIES")
    print("-" * 70)

    generated_queries = generate_queries(
        question=QUESTION,
        query_count=QUERY_COUNT,
    )

    queries = [QUESTION]

    for query in generated_queries:
        if query.strip().lower() != QUESTION.strip().lower():
            queries.append(query)

    for index, query in enumerate(queries, start=1):
        print()
        print(f"Query {index}:")
        print(query)

    # ------------------------------------------------------------------
    # 2. Retrieve original question separately
    # ------------------------------------------------------------------

    print()
    print("-" * 70)
    print("RETRIEVAL")
    print("-" * 70)

    print()
    print("Searching original question...")
    original_documents = retrieve_documents(
        question=QUESTION,
        k=RETRIEVAL_K,
    )

    generated_documents_by_query = []

    for query in queries[1:]:
        print()
        print(f"Searching generated query...")
        print(query)

        documents = retrieve_documents(
            question=query,
            k=RETRIEVAL_K,
        )

        generated_documents_by_query.append(documents)

    # ------------------------------------------------------------------
    # 3. Weighted + lexical ranking
    # ------------------------------------------------------------------

    ranked_documents = weighted_rank_documents(
        original_documents=original_documents,
        documents_by_query=generated_documents_by_query,
        original_question=QUESTION,
        original_weight=2.0,
        lexical_weight=2.0,
    )

    # ------------------------------------------------------------------
    # 4. Deduplicate
    # ------------------------------------------------------------------

    unique_documents = []
    seen_keys = set()

    for document in ranked_documents:
        key = _document_key(document)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_documents.append(document)

    # ------------------------------------------------------------------
    # 5. Recalculate diagnostic scores
    # ------------------------------------------------------------------

    scores = {}

    # Original-query contribution
    original_count = max(len(original_documents), 1)

    for rank, document in enumerate(original_documents, start=1):
        key = _document_key(document)

        original_score = (
            (original_count - rank + 1)
            / original_count
        )

        scores[key] = scores.get(key, 0.0) + (
            2.0 * original_score
        )

    # Generated-query RRF contribution
    for query_documents in generated_documents_by_query:
        for rank, document in enumerate(query_documents, start=1):
            key = _document_key(document)

            generated_score = 1.0 / (60 + rank)

            scores[key] = scores.get(key, 0.0) + generated_score

    # Lexical contribution
    for document in unique_documents:
        key = _document_key(document)

        lexical_score = lexical_relevance_score(
            QUESTION,
            document,
        )

        scores[key] = scores.get(key, 0.0) + (
            2.0 * lexical_score
        )

    # ------------------------------------------------------------------
    # 6. Print final ranking
    # ------------------------------------------------------------------

    print()
    print("-" * 70)
    print("FINAL V3 WEIGHTED + LEXICAL RANKING")
    print("-" * 70)

    print()
    print(f"Documents after deduplication: {len(unique_documents)}")

    for rank, document in enumerate(unique_documents, start=1):
        key = _document_key(document)

        print_document(
            rank=rank,
            document=document,
            score=scores.get(key, 0.0),
        )

        lexical_score = lexical_relevance_score(
            QUESTION,
            document,
        )

        print(
            f"Lexical relevance: {lexical_score:.6f}"
        )
        print()

    # ------------------------------------------------------------------
    # 7. Explicit success check for Q037
    # ------------------------------------------------------------------

    correct_rank = None

    for rank, document in enumerate(unique_documents, start=1):
        content = document.page_content.lower()

        if (
            "first step" in content
            and "self-assessment" in content
            and "career goals" in content
            and "target roles" in content
        ):
            correct_rank = rank
            break

    print("-" * 70)
    print("Q037 CHECK")
    print("-" * 70)

    if correct_rank is None:
        print("FAIL: Correct Q037 chunk was not retrieved.")
    else:
        print(
            f"Correct Q037 chunk found at final rank: "
            f"{correct_rank}"
        )

        if correct_rank <= 4:
            print("PASS: Correct chunk is in the top 4.")
        else:
            print("FAIL: Correct chunk is below the top 4.")

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()