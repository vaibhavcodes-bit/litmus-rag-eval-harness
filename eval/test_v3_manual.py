import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.multi_query import (
    generate_queries,
    retrieve_documents,
)


QUESTION = "What is the recommended first step before starting a job search?"


def main():
    print("\n" + "=" * 70)
    print("V3 MULTI-QUERY RETRIEVAL DEBUG")
    print("=" * 70)

    print("\nOriginal question:")
    print(QUESTION)

    # ---------------------------------------------------------
    # 1. Generate queries
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("GENERATED QUERIES")
    print("-" * 70)

    queries = generate_queries(
        question=QUESTION,
        query_count=4,
    )

    for index, query in enumerate(queries, start=1):
        print(f"\nQuery {index}:")
        print(query)

    # ---------------------------------------------------------
    # 2. Test each query separately
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("INDIVIDUAL QUERY RETRIEVAL")
    print("-" * 70)

    for query_index, query in enumerate(queries, start=1):

        print(f"\n{'=' * 70}")
        print(f"QUERY {query_index}")
        print(f"{'=' * 70}")
        print(query)

        documents = retrieve_documents(
            question=query,
            k=4,
        )

        for rank, document in enumerate(documents, start=1):
            source = document.metadata.get("source")
            chunk_id = document.metadata.get("chunk_id")

            print(f"\nRank {rank}")
            print(f"Source: {source}")
            print(f"Chunk ID: {chunk_id}")
            print("Content:")
            print(document.page_content[:400])

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()