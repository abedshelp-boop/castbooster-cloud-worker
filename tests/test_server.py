# tests/test_server.py
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_healthz_returns_200_and_status_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
