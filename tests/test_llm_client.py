from src.generation.llm_client import generate_answer


def test_generate_answer():
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


def test_generate_answer_refuses_unknown_information():
    context = """
    Company working hours are from 9 AM to 6 PM.
    """

    answer = generate_answer(
        question="How many vacation days do employees receive?",
        context=context,
    )

    assert answer
    assert isinstance(answer, str)