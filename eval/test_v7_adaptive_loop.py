from src.graph import adaptive_rag_graph


def test_v7_answer_retry_loop(monkeypatch):
    """
    Deterministically prove that V7 can:

    1. Generate an answer.
    2. Receive BAD from the answer grader.
    3. Retry retrieval.
    4. Rewrite the query.
    5. Generate again.
    6. Receive GOOD.
    7. Finish successfully.
    """

    # --------------------------------------------------------
    # Fake answer grades
    # --------------------------------------------------------
    #
    # First generated answer -> BAD
    # Second generated answer -> GOOD
    #

    grades = iter(["BAD", "GOOD"])

    def fake_grade_answer(question, context, answer):
        return next(grades)

    monkeypatch.setattr(
        adaptive_rag_graph,
        "grade_answer",
        fake_grade_answer,
    )

    # --------------------------------------------------------
    # Fake query rewriting
    # --------------------------------------------------------
    #
    # The rewrite should happen on the second VECTOR
    # retrieval attempt.
    #

    rewritten_queries = []

    def fake_rewrite_query(question):
        rewritten = (
            "resume summary key elements "
            "experience achievements skills"
        )

        rewritten_queries.append(rewritten)

        return rewritten

    monkeypatch.setattr(
        adaptive_rag_graph,
        "rewrite_query",
        fake_rewrite_query,
    )

    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    app = adaptive_rag_graph.build_graph()

    # --------------------------------------------------------
    # IMPORTANT:
    # This must be a VECTOR question.
    #
    # "How many remote jobs are available?"
    # routes to SQL and therefore does not increment
    # retrieval_attempts.
    #
    # This question routes to VECTOR and exercises:
    #
    # VECTOR -> retrieve -> generate -> BAD
    #        -> retry -> rewrite -> retrieve -> generate
    #        -> GOOD
    # --------------------------------------------------------

    result = app.invoke(
        {
            "question": "What should a resume summary contain?"
        }
    )

    # --------------------------------------------------------
    # Verify final answer grade
    # --------------------------------------------------------

    assert result["answer_grade"] == "GOOD"

    # --------------------------------------------------------
    # Verify two answer-generation attempts
    # --------------------------------------------------------

    assert result["answer_attempts"] == 2

    # --------------------------------------------------------
    # Verify two VECTOR retrieval attempts
    # --------------------------------------------------------

    assert result["retrieval_attempts"] == 2

    # --------------------------------------------------------
    # Verify query rewriting happened
    # --------------------------------------------------------

    assert len(rewritten_queries) == 1

    assert rewritten_queries[0] == (
        "resume summary key elements "
        "experience achievements skills"
    )

    # --------------------------------------------------------
    # Verify rewritten query was stored in state
    # --------------------------------------------------------

    assert result["rewritten_queries"] == [
        (
            "resume summary key elements "
            "experience achievements skills"
        )
    ]