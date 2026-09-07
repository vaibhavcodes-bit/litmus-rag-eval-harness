import pytest

from src.retrieval.router import (
    _clean_router_output,
    route_question,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, prompt):
        return FakeResponse(self.response)


def test_clean_router_output_vector():
    assert _clean_router_output("VECTOR") == "VECTOR"


def test_clean_router_output_sql():
    assert _clean_router_output("SQL") == "SQL"


def test_clean_router_output_handles_markdown():
    assert _clean_router_output("**VECTOR**") == "VECTOR"


def test_clean_router_output_handles_extra_text():
    assert _clean_router_output("The correct route is SQL.") == "SQL"


def test_clean_router_output_rejects_invalid_output():
    with pytest.raises(ValueError):
        _clean_router_output("WEB")


def test_route_question_rejects_empty_question():
    with pytest.raises(ValueError, match="question must not be empty"):
        route_question("")


def test_route_question_rejects_whitespace_question():
    with pytest.raises(ValueError, match="question must not be empty"):
        route_question("   ")


def test_route_question_vector(monkeypatch):
    fake_llm = FakeLLM("VECTOR")

    monkeypatch.setattr(
        "src.retrieval.router.get_router_llm",
        lambda: fake_llm,
    )

    result = route_question(
        "What should a resume summary contain?"
    )

    assert result == "VECTOR"


def test_route_question_sql(monkeypatch):
    fake_llm = FakeLLM("SQL")

    monkeypatch.setattr(
        "src.retrieval.router.get_router_llm",
        lambda: fake_llm,
    )

    result = route_question(
        "How many remote jobs are available?"
    )

    assert result == "SQL"


def test_route_question_handles_llm_list_content(monkeypatch):
    fake_llm = FakeLLM(["VECTOR"])

    monkeypatch.setattr(
        "src.retrieval.router.get_router_llm",
        lambda: fake_llm,
    )

    result = route_question(
        "What makes a cover letter effective?"
    )

    assert result == "VECTOR"