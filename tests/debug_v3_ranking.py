import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Project path setup
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from src.retrieval.multi_query import (
    generate_queries,
    debug_rank_documents,
    rerank_by_original_question,
)

from src.retrieval.retriever import retrieve_documents


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

QUESTION = "What is the recommended first step before starting a job search?"

RETRIEVAL_K = 4
QUERY_COUNT = 4


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

print("=" * 70)
print("V3 RANKING DIAGNOSTIC")
print("=" * 70)

print()
print("Original question:")
print(QUESTION)


# ----------------------------------------------------------------------
# Generate queries
# ----------------------------------------------------------------------

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

    if query.strip().lower() == QUESTION.strip().lower():
        continue

    queries.append(query)


for index, query in enumerate(queries, start=1):

    print()
    print(f"Query {index}:")
    print(query)


# ----------------------------------------------------------------------
# Retrieve documents
# ----------------------------------------------------------------------

print()
print("-" * 70)
print("RETRIEVAL")
print("-" * 70)

documents_by_query = []


for index, query in enumerate(queries, start=1):

    print()
    print(f"Searching Query {index}...")
    print(query)

    documents = retrieve_documents(
        question=query,
        k=RETRIEVAL_K,
    )

    documents_by_query.append(documents)


# ----------------------------------------------------------------------
# RRF ranking
# ----------------------------------------------------------------------

print()
print("-" * 70)
print("RRF SCORES")
print("-" * 70)

debug_results = debug_rank_documents(
    documents_by_query=documents_by_query,
)


for rank, result in enumerate(debug_results, start=1):

    print()
    print("=" * 70)
    print(f"RANK {rank}")
    print("=" * 70)

    print(f"Source: {result['source']}")
    print(f"RRF Score: {result['rrf_score']:.6f}")

    print()
    print("Content:")

    print(result["content"])

    print()
    print("Appearances:")

    for appearance in result["appearances"]:

        print(
            f"  Query {appearance['query']} "
            f"-> Rank {appearance['rank']} "
            f"-> Score {appearance['score']:.6f}"
        )


# ----------------------------------------------------------------------
# Convert RRF results back into Documents
# ----------------------------------------------------------------------

documents_by_key = {}

for query_documents in documents_by_query:

    for document in query_documents:

        chunk_id = document.metadata.get("chunk_id")

        if chunk_id:
            documents_by_key[chunk_id] = document


rrf_documents = []

for result in debug_results:

    document_key = result["document_key"]

    document = documents_by_key.get(document_key)

    if document is not None:
        rrf_documents.append(document)


# ----------------------------------------------------------------------
# Original-question reranking
# ----------------------------------------------------------------------

print()
print("-" * 70)
print("ORIGINAL QUESTION RERANKING")
print("-" * 70)

reranked_documents = rerank_by_original_question(
    documents=rrf_documents,
    question=QUESTION,
)


# ----------------------------------------------------------------------
# Print reranked documents
# ----------------------------------------------------------------------

for rank, document in enumerate(
    reranked_documents,
    start=1,
):

    print()
    print("=" * 70)
    print(f"RERANKED RANK {rank}")
    print("=" * 70)

    print(
        f"Source: "
        f"{document.metadata.get('source')}"
    )

    print(
        f"Chunk ID: "
        f"{document.metadata.get('chunk_id')}"
    )

    print()
    print("Content:")

    print(
        document.page_content[:500]
        .replace("\n", " ")
    )


# ----------------------------------------------------------------------
# Diagnostic complete
# ----------------------------------------------------------------------

print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)