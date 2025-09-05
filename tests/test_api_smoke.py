"""Smoke tests for FastAPI app."""

from fastapi.testclient import TestClient

from src.serve.api import app


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


