from src.pipeline import answer_question


def test_v4_pipeline_comparison_question():
    result = answer_question(
        "Compare resume summary and resume objective.",
        k=4,
        mode="v4",
    )

    assert "answer" in result
    assert "sources" in result
    assert "sub_questions" in result

    assert len(result["sub_questions"]) >= 2

    assert result["answer"]

    assert len(result["sources"]) > 0