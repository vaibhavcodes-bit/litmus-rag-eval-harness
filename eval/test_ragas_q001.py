import os
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)
from src.retrieval.embedder import get_embeddings

def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    # Groq's OpenAI-compatible API
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Ragas evaluator LLM
    ragas_llm = llm_factory(
        model="openai/gpt-oss-20b",
        provider="openai",
        client=client,
    )

    # One test question from the golden dataset
    dataset = Dataset.from_list(
    [
        {
            "user_input": (
                "What is the ideal length for a resume "
                "with under 10 years of experience?"
            ),
            "response": "One page.",
            "reference": "One page.",
            "retrieved_contexts": [
                (
                    "Candidates with fewer than 10 years of "
                    "professional experience should generally "
                    "keep their resume to one page."
                )
            ],
        }
    ]
)

    # Ragas generation metrics
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        AnswerCorrectness(llm=ragas_llm),
    ]

    embeddings = get_embeddings()

    result = evaluate(
            dataset=dataset,
            metrics=metrics,
            embeddings=embeddings,
            raise_exceptions=True,
            show_progress=True,
        )

    print("\n" + "=" * 50)
    print("Ragas Q001 Result")
    print("=" * 50)
    print(result)


if __name__ == "__main__":
    main()