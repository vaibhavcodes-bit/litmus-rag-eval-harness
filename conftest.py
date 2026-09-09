"""
Pytest configuration file with fixtures for testing.
This file handles mocking of Groq API calls for unit tests.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_groq_client():
    """
    Mock the Groq ChatGroq client to avoid actual API calls during testing.
    This fixture prevents the "Illegal header value" error caused by 
    invalid API key formatting in HTTP headers.
    """
    with patch('src.generation.llm_client.ChatGroq') as mock:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(
            content="This is a mocked response from the LLM."
        )
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_groq_client_with_context():
    """
    Mock the Groq ChatGroq client with context-aware responses.
    Useful for testing that the LLM correctly processes context.
    """
    with patch('src.generation.llm_client.ChatGroq') as mock:
        mock_instance = MagicMock()
        
        def mock_invoke(prompt):
            if "company working hours" in prompt.lower():
                return MagicMock(content="Company working hours are from 9 AM to 6 PM.")
            elif "vacation days" in prompt.lower():
                return MagicMock(content="I don't have enough information to answer that.")
            else:
                return MagicMock(content="General mocked response.")
        
        mock_instance.invoke.side_effect = mock_invoke
        mock.return_value = mock_instance
        yield mock
