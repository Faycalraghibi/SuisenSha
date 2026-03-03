from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pipeline.api.main import app

# Ensure API tests do not load Heavy ML artifacts unless specifically triggered by the user in dev mode to save testing time on CI.
client = TestClient(app)

def test_get_user_history_missing_user():
    response = client.get("/users/99999/history")
    assert response.status_code == 404
    assert response.json() == {"detail": "User not found in evaluation set."}

def test_api_docs_available():
    # Swagger docs load without needing ML models active
    response = client.get("/docs")
    assert response.status_code == 200

# Testing the ML endpoints requires the lifespan context manager to run.
# Since pytest-fastapi `TestClient` doesn't always run lifespan triggers automatically natively,
# we would orchestrate this with `with TestClient(app) as client:` if testing the heavy models in CI.
# For now, these basic tests verify routing and error handling.
