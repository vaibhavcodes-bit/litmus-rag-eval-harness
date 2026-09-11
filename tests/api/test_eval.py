from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_latest_evaluation():
    response = client.get("/eval/latest")

    assert response.status_code == 200

    data = response.json()

    assert "version" in data
    assert "summary" in data
    assert "questions" in data

    assert isinstance(data["version"], str)
    assert isinstance(data["summary"], dict)
    assert isinstance(data["questions"], list)

    assert data["summary"]
    assert data["questions"]