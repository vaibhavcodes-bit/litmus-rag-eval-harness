from src.retrieval.sql_source import (
    average_salary,
    count_jobs,
    get_job_by_id,
    search_jobs,
)


def test_count_all_jobs():
    assert count_jobs() == 20


def test_count_remote_jobs():
    assert count_jobs(remote=True) == 14


def test_count_non_remote_jobs():
    assert count_jobs(remote=False) == 6


def test_count_senior_jobs():
    assert count_jobs(experience_level="senior") == 7


def test_count_bangalore_jobs():
    assert count_jobs(location="Bangalore") == 7


def test_combined_filters():
    result = count_jobs(
        location="Bangalore",
        remote=True,
        experience_level="senior",
    )

    assert result == 4


def test_average_senior_salary():
    result = average_salary(
        experience_level="senior"
    )

    assert round(
        result["average_salary_min"],
        2,
    ) == 2028571.43

    assert round(
        result["average_salary_max"],
        2,
    ) == 3185714.29


def test_search_remote_jobs():
    results = search_jobs(
        remote=True,
        limit=5,
    )

    assert len(results) == 5

    for job in results:
        assert job["remote"] == 1


def test_search_bangalore_jobs():
    results = search_jobs(
        location="Bangalore",
        limit=10,
    )

    assert len(results) == 7

    for job in results:
        assert job["location"] == "Bangalore"


def test_search_senior_remote_jobs():
    results = search_jobs(
        remote=True,
        experience_level="senior",
        limit=10,
    )

    assert len(results) == 7

    for job in results:
        assert job["remote"] == 1
        assert job["experience_level"] == "senior"


def test_search_title():
    results = search_jobs(
        title="Backend",
        limit=10,
    )

    assert len(results) == 3

    for job in results:
        assert "Backend" in job["title"]


def test_get_existing_job():
    job = get_job_by_id(1)

    assert job is not None
    assert job["id"] == 1
    assert job["title"] == "Backend Engineer"
    assert job["company"] == "TechNova"


def test_get_missing_job():
    job = get_job_by_id(9999)

    assert job is None


def test_search_limit():
    results = search_jobs(limit=3)

    assert len(results) == 3


def test_invalid_limit_zero():
    try:
        search_jobs(limit=0)
        assert False
    except ValueError as exc:
        assert str(exc) == "limit must be at least 1"


def test_invalid_limit_too_large():
    try:
        search_jobs(limit=101)
        assert False
    except ValueError as exc:
        assert str(exc) == "limit cannot exceed 100"