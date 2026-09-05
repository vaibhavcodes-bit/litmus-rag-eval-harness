from src.pipeline import answer_question


def main():
    print("=" * 60)
    print("Litmus V1 - RAG CLI")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = answer_question(question)

            print("\nAnswer:")
            print(result["answer"])

            print("\nSources:")

            if not result["sources"]:
                print("No sources found.")
                continue

            for source in result["sources"]:
                print(
                    f"- {source['source']}, "
                    f"page {source['page'] + 1}"
                )

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()