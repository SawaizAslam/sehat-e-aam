"""LLM client abstraction.

Supports two backends:

1. ``databricks``: Databricks Foundation Model API via ``databricks-sdk``.
2. ``openai``: any OpenAI-compatible HTTP endpoint (Ollama, Together, Groq,
   OpenAI, vLLM, ...).

Both paths request JSON-mode output where the underlying model supports it
(``response_format={"type": "json_object"}``), eliminating the markdown-fence
parsing hack in the original guide.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""

    def parse_json(self) -> dict[str, Any]:
        """Parse the content as JSON, tolerantly handling markdown fences."""

        cleaned = self.content.strip()
        if cleaned.startswith("```"):
            # remove leading fence and optional language tag
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()
        return json.loads(cleaned)


class LLMError(RuntimeError):
    """Raised on non-retryable LLM failures."""


class _RetryableLLMError(RuntimeError):
    """Internal: triggers tenacity retry."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _OpenAIBackend:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # local import avoids hard dep at module load

        self._client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        self._model = settings.llm_model

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        import time

        from openai import APIConnectionError, APIError, RateLimitError

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError) as e:
            raise _RetryableLLMError(str(e)) from e
        except APIError as e:
            status = getattr(e, "status_code", 0)
            if status in {408, 409, 429, 500, 502, 503, 504}:
                raise _RetryableLLMError(str(e)) from e
            raise LLMError(f"OpenAI-compatible API error: {e}") from e
        latency_ms = (time.time() - start) * 1000

        choice = resp.choices[0]
        content = choice.message.content or ""
        usage = resp.usage
        return LLMResponse(
            content=content,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            model=self._model,
        )


class _DatabricksBackend:
    def __init__(self, settings: Settings) -> None:
        from databricks.sdk import WorkspaceClient

        kwargs: dict[str, Any] = {}
        if settings.databricks_host:
            kwargs["host"] = settings.databricks_host
        if settings.databricks_token:
            kwargs["token"] = settings.databricks_token
        self._client = WorkspaceClient(**kwargs)
        self._model = settings.llm_model

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        import time

        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        sdk_messages = [
            ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"])
            for m in messages
        ]
        query_kwargs: dict[str, Any] = {
            "name": self._model,
            "messages": sdk_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Databricks Foundation Model API accepts response_format via extra_params
            query_kwargs["extra_params"] = {"response_format": {"type": "json_object"}}

        start = time.time()
        try:
            try:
                resp = self._client.serving_endpoints.query(**query_kwargs)
            except TypeError:
                # Older SDK versions do not accept ``extra_params``; retry without it.
                query_kwargs.pop("extra_params", None)
                resp = self._client.serving_endpoints.query(**query_kwargs)
        except Exception as e:  # pragma: no cover - depends on remote
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "timeout" in msg or "503" in msg:
                raise _RetryableLLMError(str(e)) from e
            raise LLMError(f"Databricks serving error: {e}") from e
        latency_ms = (time.time() - start) * 1000

        # databricks-sdk returns a dict-like object; extract content defensively
        choices = resp.choices or []
        content = ""
        if choices:
            msg = choices[0].message
            content = (msg.content if msg else "") or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        if json_mode:
            content = content.strip()

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model=self._model,
        )


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.llm_backend == "databricks":
            self._backend: Any = _DatabricksBackend(self.settings)
        else:
            self._backend = _OpenAIBackend(self.settings)

    @retry(
        retry=retry_if_exception_type(_RetryableLLMError),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ) -> LLMResponse:
        return self._backend.complete(
            messages=messages,
            temperature=temperature
            if temperature is not None
            else self.settings.extract_temperature,
            max_tokens=max_tokens or self.settings.extract_max_tokens,
            json_mode=json_mode,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LLMResponse]:
        """Call the LLM and parse JSON. Raises ``LLMError`` if parsing fails."""

        resp = self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        try:
            data = resp.parse_json()
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned non-JSON content: {e}; raw={resp.content[:200]!r}") from e
        return data, resp


__all__ = ["LLMClient", "LLMResponse", "LLMError"]
