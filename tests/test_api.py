from __future__ import annotations

from fastapi.testclient import TestClient

from pipeline.api.main import app

client = TestClient(app)


def test_get_user_history_missing_user():
    response = client.get("/users/99999/history")
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found in evaluation set."}


def test_api_docs_available():
    response = client.get("/docs")
    assert response.status_code == 200
