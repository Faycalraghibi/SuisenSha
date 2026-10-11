from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import requests

from pipeline.config import rag_cfg

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract interface for LLM text generation backends."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text given a prompt string."""


class FakeLLMBackend(LLMBackend):
    """Deterministic fake backend for fast tests without downloading models."""

    def __init__(self, canned_response: str = "[]", responses: list[str] | None = None) -> None:
        self.canned_response = canned_response
        self.responses = list(responses) if responses is not None else []
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        self._call_count += 1
        if self.responses:
            idx = (self._call_count - 1) % len(self.responses)
            return self.responses[idx]
        return self.canned_response


class LocalHFBackend(LLMBackend):
    """Local HuggingFace model backend with lazy loading."""

    def __init__(
        self,
        model_name: str = rag_cfg.llm_model_name,
        max_new_tokens: int = rag_cfg.max_new_tokens,
        device: int = -1,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._pipeline: Any = None

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline

        logger.info("Loading local LLM: %s ...", self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
        )
        logger.info("Local LLM %s loaded successfully.", self.model_name)
        return self._pipeline

    def generate(self, prompt: str) -> str:
        gen = self._load_pipeline()
        output = gen(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=gen.tokenizer.eos_token_id,
        )
        text = output[0]["generated_text"]
        if text.startswith(prompt):
            return text[len(prompt) :].strip()
        return text.strip()


class HostedAPIBackend(LLMBackend):
    """Hosted OpenAI-compatible API backend (e.g. OpenAI, OpenRouter, Groq, vLLM)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ) -> None:
        raw_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.base_url = (raw_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


_backend_singleton: LLMBackend | None = None


def get_llm_backend(backend_type: str | None = None) -> LLMBackend | None:
    """Factory to get or instantiate the requested LLM backend."""
    global _backend_singleton
    raw_selected = backend_type or getattr(rag_cfg, "llm_backend", "local")
    selected = str(raw_selected if raw_selected is not None else "local").lower()

    if selected in ("disabled", "none", "off"):
        return None

    if selected == "fake":
        return FakeLLMBackend()

    if selected == "hosted":
        return HostedAPIBackend(
            base_url=getattr(rag_cfg, "api_base_url", None),
            api_key=getattr(rag_cfg, "api_key", None),
        )

    if _backend_singleton is None:
        _backend_singleton = LocalHFBackend()
    return _backend_singleton
