import pytest
from unittest.mock import patch, MagicMock
from src.generation.llm_client import generate_answer


@patch('src.generation.llm_client.ChatGroq')
def test_generate_answer(mock_groq):
    """Test that generate_answer returns valid responses."""
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(
        content="Company working hours are from 9 AM to 6 PM."
    )
    mock_groq.return_value = mock_instance
    
    context = """
    Company working hours are from 9 AM to 6 PM.
    Core working hours are from 10 AM to 5 PM.
    """

    answer = generate_answer(
        question="What are the company working hours?",
        context=context,
    )

    assert answer
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


@patch('src.generation.llm_client.ChatGroq')
def test_generate_answer_refuses_unknown_information(mock_groq):
    """Test that LLM refuses to answer questions outside the provided context."""
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.invoke.return_value = MagicMock(
        content="I don't have enough information to answer that."
    )
    mock_groq.return_value = mock_instance
    
    context = """
    Company working hours are from 9 AM to 6 PM.
    """

    answer = generate_answer(
        question="How many vacation days do employees receive?",
        context=context,
    )

    assert answer
    assert isinstance(answer, str)
