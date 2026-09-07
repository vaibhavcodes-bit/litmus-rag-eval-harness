import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "structured"
    / "jobs.db"
)


def get_connection() -> sqlite3.Connection:
    """
    Create a connection to the structured jobs database.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def count_jobs(
    location: str | None = None,
    remote: bool | None = None,
    experience_level: str | None = None,
) -> int:
    """
    Count jobs using optional filters.
    """

    query = "SELECT COUNT(*) FROM jobs"
    conditions = []
    parameters: list[Any] = []

    if location is not None:
        conditions.append("location = ?")
        parameters.append(location)

    if remote is not None:
        conditions.append("remote = ?")
        parameters.append(int(remote))

    if experience_level is not None:
        conditions.append("experience_level = ?")
        parameters.append(experience_level)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with get_connection() as connection:
        result = connection.execute(
            query,
            parameters,
        ).fetchone()

    return int(result[0])


def average_salary(
    experience_level: str | None = None,
    remote: bool | None = None,
    location: str | None = None,
) -> dict[str, float | int]:
    """
    Calculate average minimum and maximum salary
    using optional filters.
    """

    query = """
        SELECT
            AVG(salary_min) AS average_salary_min,
            AVG(salary_max) AS average_salary_max
        FROM jobs
    """

    conditions = []
    parameters: list[Any] = []

    if experience_level is not None:
        conditions.append("experience_level = ?")
        parameters.append(experience_level)

    if remote is not None:
        conditions.append("remote = ?")
        parameters.append(int(remote))

    if location is not None:
        conditions.append("location = ?")
        parameters.append(location)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with get_connection() as connection:
        row = connection.execute(
            query,
            parameters,
        ).fetchone()

    return {
        "average_salary_min": float(
            row["average_salary_min"]
        )
        if row["average_salary_min"] is not None
        else 0.0,
        "average_salary_max": float(
            row["average_salary_max"]
        )
        if row["average_salary_max"] is not None
        else 0.0,
    }


def search_jobs(
    title: str | None = None,
    company: str | None = None,
    location: str | None = None,
    remote: bool | None = None,
    experience_level: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search structured job records using optional filters.
    """

    if limit < 1:
        raise ValueError("limit must be at least 1")

    if limit > 100:
        raise ValueError("limit cannot exceed 100")

    query = """
        SELECT
            id,
            title,
            company,
            location,
            remote,
            experience_level,
            salary_min,
            salary_max,
            employment_type
        FROM jobs
    """

    conditions = []
    parameters: list[Any] = []

    if title is not None:
        conditions.append("title LIKE ?")
        parameters.append(f"%{title}%")

    if company is not None:
        conditions.append("company = ?")
        parameters.append(company)

    if location is not None:
        conditions.append("location = ?")
        parameters.append(location)

    if remote is not None:
        conditions.append("remote = ?")
        parameters.append(int(remote))

    if experience_level is not None:
        conditions.append("experience_level = ?")
        parameters.append(experience_level)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id LIMIT ?"
    parameters.append(limit)

    with get_connection() as connection:
        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [dict(row) for row in rows]


def get_job_by_id(job_id: int) -> dict[str, Any] | None:
    """
    Retrieve a single job by its ID.
    """

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                title,
                company,
                location,
                remote,
                experience_level,
                salary_min,
                salary_max,
                employment_type
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


if __name__ == "__main__":
    print("Database:", DB_PATH)

    print(
        "Total jobs:",
        count_jobs(),
    )

    print(
        "Remote jobs:",
        count_jobs(remote=True),
    )

    print(
        "Senior jobs:",
        count_jobs(experience_level="senior"),
    )

    print(
        "Bangalore jobs:",
        count_jobs(location="Bangalore"),
    )

    print(
        "Average senior salary:",
        average_salary(
            experience_level="senior"
        ),
    )

    print("\nFirst 3 remote jobs:")

    for job in search_jobs(
        remote=True,
        limit=3,
    ):
        print(job)