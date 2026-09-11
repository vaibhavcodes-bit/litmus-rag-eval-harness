from langchain_core.documents import Document

from src.pipeline import answer_question


def test_v5_vector_route(monkeypatch):

    documents = [
        Document(
            page_content=(
                "A resume summary should highlight "
                "skills and achievements."
            ),
            metadata={
                "source": "resume.pdf",
                "page": 1,
            },
        )
    ]

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "VECTOR",
    )

    monkeypatch.setattr(
        "src.pipeline.retrieve_documents",
        lambda question, k: documents,
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: "VECTOR ANSWER",
    )

    result = answer_question(
        "What should a resume summary contain?",
        mode="v5",
    )

    assert result["route"] == "VECTOR"

    assert result["answer"] == "VECTOR ANSWER"

    assert len(result["sources"]) == 1

    assert result["sources"][0]["source"] == "resume.pdf"


def test_v5_sql_route(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.count_jobs",
        lambda remote=True: 14,
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: "SQL ANSWER",
    )

    result = answer_question(
        "How many remote jobs are available?",
        mode="v5",
    )

    assert result["route"] == "SQL"

    assert result["answer"] == "SQL ANSWER"

    assert len(result["sources"]) == 1

    assert result["sources"][0]["source"] == "jobs.db"

    assert result["sources"][0]["type"] == "sql"


def test_v5_sql_context_contains_database_result(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.count_jobs",
        lambda remote=True: 14,
    )

    captured = {}

    def fake_generate_answer(question, context):

        captured["question"] = question
        captured["context"] = context

        return "There are 14 remote jobs."

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        fake_generate_answer,
    )

    result = answer_question(
        "How many remote jobs are available?",
        mode="v5",
    )

    assert result["route"] == "SQL"

    assert result["answer"] == (
        "There are 14 remote jobs."
    )

    assert (
        "Number of remote jobs: 14"
        in captured["context"]
    )


def test_v5_sql_senior_count(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.count_jobs",
        lambda experience_level="senior": 7,
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: (
            "There are 7 senior jobs."
        ),
    )

    result = answer_question(
        "How many senior jobs are available?",
        mode="v5",
    )

    assert result["route"] == "SQL"

    assert result["answer"] == (
        "There are 7 senior jobs."
    )

    assert (
        result["sources"][0]["query_type"]
        == "count_senior_jobs"
    )


def test_v5_sql_average_salary(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.average_salary",
        lambda experience_level=None,
        remote=None,
        location=None: {
            "average_salary_min": 2028571.43,
            "average_salary_max": 3185714.29,
        },
    )

    captured = {}

    def fake_generate_answer(question, context):

        captured["context"] = context

        return (
            "The average maximum salary "
            "for senior roles is approximately "
            "₹31.86 lakh."
        )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        fake_generate_answer,
    )

    result = answer_question(
        "What is the average maximum salary for senior roles?",
        mode="v5",
    )

    assert result["route"] == "SQL"

    assert "average_salary_max" in captured["context"]

    assert "3185714.29" in captured["context"]

    assert result["sources"][0]["query_type"] == (
        "average_salary"
    )


def test_v5_invalid_mode():

    try:

        answer_question(
            "Hello",
            mode="v9",
        )

        assert False

    except ValueError as error:

        assert (
            str(error)
            == "mode must be one of 'v1', 'v3', 'v4', 'v5', or 'v6'."
        )


def test_v5_sql_entry_level_count(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.count_jobs",
        lambda experience_level="entry": 4,
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: (
            "There are 4 entry-level jobs."
        ),
    )

    result = answer_question(
        "How many entry-level jobs are available?",
        mode="v5",
    )

    assert result["route"] == "SQL"

    assert result["answer"] == (
        "There are 4 entry-level jobs."
    )

    assert result["sources"][0]["source"] == "jobs.db"

    assert result["sources"][0]["type"] == "sql"

    assert result["sources"][0]["query_type"] == (
        "count_entry_level_jobs"
    )


def test_v5_sql_company_count(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.route_question",
        lambda question: "SQL",
    )

    monkeypatch.setattr(
        "src.pipeline.search_jobs",
        lambda company="CloudWorks", limit=100: [
            {
                "id": 2,
                "company": "CloudWorks",
            },
            {
                "id": 7,
                "company": "CloudWorks",
            },
            {
                "id": 15,
                "company": "CloudWorks",
            },
        ],
    )

    monkeypatch.setattr(
        "src.pipeline.generate_answer",
        lambda question, context: (
            "There are 3 jobs at CloudWorks."
        ),
    )

    result = answer_question(
        "How many jobs are available at CloudWorks?",
        mode="v5",
    )

    assert result["route"] == "SQL"

    assert result["answer"] == (
        "There are 3 jobs at CloudWorks."
    )

    assert result["sources"][0]["source"] == "jobs.db"

    assert result["sources"][0]["type"] == "sql"

    assert result["sources"][0]["query_type"] == (
        "count_company_jobs"
    )