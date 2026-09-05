from pathlib import Path

from src.ingest.ingest_all import RAW_DATA_DIR, ingest_all_pdfs


def test_raw_data_directory_exists():
    assert RAW_DATA_DIR.exists()
    assert RAW_DATA_DIR.is_dir()


def test_career_pdfs_exist():
    pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))

    assert len(pdf_files) == 8

    expected_files = {
        "resume_writing.pdf",
        "ats_optimization.pdf",
        "cover_letters.pdf",
        "interview_prep.pdf",
        "job_search_strategy.pdf",
        "linkedin_branding.pdf",
        "salary_negotiation.pdf",
        "remote_work_and_career_pivot.pdf",
    }

    actual_files = {pdf.name for pdf in pdf_files}

    assert actual_files == expected_files


def test_ingest_all_pdfs(monkeypatch):
    calls = []

    def fake_add_documents(documents):
        calls.append(documents)

    monkeypatch.setattr(
        "src.ingest.ingest_all.add_documents",
        fake_add_documents,
    )

    ingest_all_pdfs()

    assert len(calls) == 8

    total_chunks = sum(
        len(documents)
        for documents in calls
    )

    assert total_chunks > 0