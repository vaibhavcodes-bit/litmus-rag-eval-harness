import csv
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "structured" / "jobs.csv"
DB_PATH = PROJECT_ROOT / "data" / "structured" / "jobs.db"


def create_database():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                location TEXT NOT NULL,
                remote INTEGER NOT NULL,
                experience_level TEXT NOT NULL,
                salary_min INTEGER NOT NULL,
                salary_max INTEGER NOT NULL,
                employment_type TEXT NOT NULL
            )
            """
        )

        with CSV_PATH.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            rows = [
                (
                    int(row["id"]),
                    row["title"],
                    row["company"],
                    row["location"],
                    int(row["remote"]),
                    row["experience_level"],
                    int(row["salary_min"]),
                    int(row["salary_max"]),
                    row["employment_type"],
                )
                for row in reader
            ]

        cursor.executemany(
            """
            INSERT INTO jobs (
                id,
                title,
                company,
                location,
                remote,
                experience_level,
                salary_min,
                salary_max,
                employment_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        connection.commit()

        cursor.execute("SELECT COUNT(*) FROM jobs")
        count = cursor.fetchone()[0]

        print(f"Database created: {DB_PATH}")
        print(f"Rows inserted: {count}")

        cursor.execute(
            """
            SELECT
                id,
                title,
                company,
                location,
                remote,
                experience_level,
                salary_min,
                salary_max
            FROM jobs
            ORDER BY id
            LIMIT 5
            """
        )

        print("\nFirst 5 jobs:")

        for job in cursor.fetchall():
            print(job)

    finally:
        connection.close()


if __name__ == "__main__":
    create_database()