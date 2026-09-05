from eval.run_eval import (
    load_dataset,
    evaluate_question,
)


def test_load_dataset():
    dataset = load_dataset()

    assert isinstance(dataset, dict)
    assert dataset["total_questions"] == 50
    assert len(dataset["questions"]) == 50


def test_evaluate_question(monkeypatch):
    fake_documents = [
        {
            "source": "wrong.pdf",
            "page": 0,
            "content": "Some unrelated information.",
        },
        {
            "source": "resume_writing.pdf",
            "page": 0,
            "content": "Use measurable achievements in your resume.",
        },
    ]

    def fake_retrieve_documents(question, k):
        return [
            type(
                "FakeDocument",
                (),
                {
                    "metadata": {
                        "source": document["source"],
                        "page": document["page"],
                    },
                    "page_content": document["content"],
                },
            )()
            for document in fake_documents
        ]

    monkeypatch.setattr(
        "eval.run_eval.retrieve_documents",
        fake_retrieve_documents,
    )

    question_data = {
        "id": "Q001",
        "question": "How can I make my resume stronger?",
        "category": "Resume Writing",
        "type": "factual",
        "difficulty": "easy",
        "reference_document": "resume_writing.pdf",
        "ground_truth": "Use measurable achievements.",
    }

    result = evaluate_question(
        question_data=question_data,
        k=2,
    )

    assert result["id"] == "Q001"
    assert result["question"] == "How can I make my resume stronger?"
    assert result["expected_document"] == "resume_writing.pdf"

    assert len(result["retrieved_documents"]) == 2

    assert result["hit_rate_at_k"] == 1.0
    assert result["mrr"] == 0.5
    assert result["context_precision"] == 0.5


def test_evaluate_unanswerable_question(monkeypatch):
    def fake_retrieve_documents(question, k):
        return [
            type(
                "FakeDocument",
                (),
                {
                    "metadata": {
                        "source": "resume_writing.pdf",
                        "page": 0,
                    },
                    "page_content": "Resume writing information.",
                },
            )()
        ]

    monkeypatch.setattr(
        "eval.run_eval.retrieve_documents",
        fake_retrieve_documents,
    )

    question_data = {
        "id": "Q050",
        "question": "What is the capital of France?",
        "category": "Out of Scope",
        "type": "unanswerable",
        "difficulty": "hard",
        "reference_document": None,
        "ground_truth": None,
    }

    result = evaluate_question(
        question_data=question_data,
        k=4,
    )

    assert result["id"] == "Q050"
    assert result["expected_document"] is None

    assert result["hit_rate_at_k"] == 0.0
    assert result["mrr"] == 0.0
    assert result["context_precision"] == 0.0