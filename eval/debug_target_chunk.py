import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.retriever import retrieve_documents


QUESTION = (
    "self-assessment career goals target roles "
    "target industries non-negotiables"
)


def main():
    print("\n" + "=" * 70)
    print("TARGET CHUNK RETRIEVAL TEST")
    print("=" * 70)

    print("\nQuery:")
    print(QUESTION)

    documents = retrieve_documents(
        question=QUESTION,
        k=10,
    )

    print(f"\nRetrieved documents: {len(documents)}")

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source")
        chunk_id = document.metadata.get("chunk_id")

        print("\n" + "-" * 70)
        print(f"Rank: {index}")
        print(f"Source: {source}")
        print(f"Chunk ID: {chunk_id}")
        print("-" * 70)

        print(document.page_content[:700])


if __name__ == "__main__":
    main()