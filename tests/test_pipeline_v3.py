from src.pipeline import answer_question


def test_pipeline_supports_v3_multi_query_mode():
    question = "What is the recommended first step before starting a job search?"

    result = answer_question(
        question=question,
        mode="v3",
    )

    assert result
    assert result["answer"]
    assert "sources" in result
    assert result["sources"]

    combined_sources = "\n".join(
        source["content"].lower()
        for source in result["sources"]
    )

    assert "self-assessment" in combined_sources
    assert "career goals" in combined_sources