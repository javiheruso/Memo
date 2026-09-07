from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "memo"
    assert body["environment"] == "development"


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["app"] == "memo"
