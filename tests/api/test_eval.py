from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_latest_evaluation():
    response = client.get("/eval/latest")

    assert response.status_code == 200

    data = response.json()

    assert data["version"] == "v6"
    assert "summary" in data
    assert "questions" in data

    assert data["summary"]["question_count"] == 10
    assert data["summary"]["hit_rate_at_4"] == 1.0
    assert data["summary"]["mrr"] == 1.0
    assert data["summary"]["context_precision"] == 1.0

    assert len(data["questions"]) == 10