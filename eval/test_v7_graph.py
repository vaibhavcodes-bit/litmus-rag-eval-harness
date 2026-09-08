from src.graph.adaptive_rag_graph import build_graph


def test_v7_vector_end_to_end():
    """
    V7 VECTOR:
    Route -> Retrieve -> Grade -> Generate -> Grade Answer -> END
    """

    app = build_graph()

    result = app.invoke(
        {
            "question": "What should a resume summary contain?"
        }
    )

    assert result["route"] == "VECTOR"

    assert result["relevant"] is True

    assert result["answer"]

    assert result["answer_grade"] == "GOOD"

    assert result["answer_attempts"] >= 1

    assert result["retrieval_attempts"] >= 1

    assert len(result["sources"]) >= 1

    assert any(
        source["source"] == "resume_writing.pdf"
        for source in result["sources"]
    )


def test_v7_sql_end_to_end():
    """
    V7 SQL:
    Route -> SQL -> Generate -> Grade Answer -> END
    """

    app = build_graph()

    result = app.invoke(
        {
            "question": "How many remote jobs are available?"
        }
    )

    assert result["route"] == "SQL"

    assert result["sql_result"] == 14

    assert result["context"] == (
        "Number of remote jobs: 14"
    )

    assert result["answer"]

    assert result["answer_grade"] == "GOOD"

    assert result["answer_attempts"] == 1

    assert result["retrieval_attempts"] == 0

    assert result["sources"]

    assert result["sources"][0]["source"] == "jobs.db"