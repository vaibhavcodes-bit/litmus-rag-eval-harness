from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_ask_v3():
    response = client.post(
        "/ask",
        json={
            "question": "What is the recommended first step before starting a job search?",
            "k": 4,
            "mode": "v3",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]
    assert "sources" in data
    assert data["sources"]

    combined_sources = "\n".join(
        source["content"].lower()
        for source in data["sources"]
    )

    assert "self-assessment" in combined_sources
    assert "career goals" in combined_sources