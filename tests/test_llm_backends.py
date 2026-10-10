from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.models.llm import (
    FakeLLMBackend,
    HostedAPIBackend,
    LLMBackend,
    get_llm_backend,
)


def test_fake_llm_backend():
    backend = FakeLLMBackend(canned_response='[{"item_id": 1, "reason": "Because of action."}]')
    assert isinstance(backend, LLMBackend)
    result = backend.generate("test prompt")
    assert result == '[{"item_id": 1, "reason": "Because of action."}]'


def test_hosted_api_backend_success(monkeypatch: pytest.MonkeyPatch):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '[{"item_id": 10, "reason": "Grounded reason."}]'}}]
    }

    with patch("requests.post", return_value=mock_response) as mock_post:
        backend = HostedAPIBackend(
            base_url="https://api.example.com/v1",
            api_key="secret-key",
            model="gpt-4o-mini",
        )
        out = backend.generate("Recommend movies")
        assert out == '[{"item_id": 10, "reason": "Grounded reason."}]'
        mock_post.assert_called_once()


def test_get_llm_backend_factory():
    fake = get_llm_backend("fake")
    assert isinstance(fake, FakeLLMBackend)
