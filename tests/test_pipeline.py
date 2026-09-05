from src.pipeline import answer_question


def test_answer_question():
    result = answer_question(
        "What are the company working hours?"
    )

    assert isinstance(result, dict)

    assert "answer" in result
    assert "sources" in result

    assert result["answer"]
    assert isinstance(result["sources"], list)

    assert len(result["sources"]) > 0

    first_source = result["sources"][0]

    assert "source" in first_source
    assert "page" in first_source
    assert "content" in first_source

    assert first_source["content"].strip() != ""