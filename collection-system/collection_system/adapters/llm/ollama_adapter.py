"""Ollama local LLM adapter — fallback when Groq is unavailable."""
from __future__ import annotations

import json
import re

import httpx
import structlog

from collection_system.agents.prompts import (
    EXPAND_TOPIC_SYSTEM,
    EXPAND_TOPIC_USER,
    SCORE_RELEVANCE_SYSTEM,
    SCORE_RELEVANCE_USER,
)
from collection_system.core.errors import LLMError

log = structlog.get_logger()

DEFAULT_MODEL = "qwen2.5:7b"
EXPAND_COUNT = 6


def _parse_json_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


class OllamaAdapter:
    """
    Ollama local inference adapter.
    Uses the OpenAI-compatible /v1/chat/completions endpoint.
    Slower than Groq but free, offline, and has no rate limit.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        max_depth: int = 3,
        timeout_s: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_depth = max_depth
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _chat(self, system: str, user: str) -> str:
        client = await self._get_client()
        try:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.4,
                    "max_tokens": 1024,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"Ollama returned non-JSON: {exc}") from exc

        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Ollama response malformed: {payload}") from exc

    async def expand_topic(
        self, topic: str, parent: str | None, depth: int
    ) -> list[str]:
        expand_from = parent if parent else topic
        user_prompt = EXPAND_TOPIC_USER.format(
            topic=expand_from,
            depth=depth,
            max_depth=self._max_depth,
            count=EXPAND_COUNT,
        )
        try:
            raw = await self._chat(EXPAND_TOPIC_SYSTEM, user_prompt)
            queries = _parse_json_array(raw)
            if not isinstance(queries, list):
                raise LLMError(f"Expected list, got {type(queries).__name__}")
            result = [str(q).strip() for q in queries if str(q).strip()]
            log.debug(
                "ollama.expand_topic",
                expand_from=expand_from,
                depth=depth,
                count=len(result),
            )
            return result
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"ollama expand_topic failed: {exc}") from exc

    async def score_relevance(self, queries: list[str], topic: str) -> list[float]:
        if not queries:
            return []
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
        user_prompt = SCORE_RELEVANCE_USER.format(topic=topic, queries=numbered)
        try:
            raw = await self._chat(SCORE_RELEVANCE_SYSTEM, user_prompt)
            scores = _parse_json_array(raw)
            if not isinstance(scores, list):
                raise LLMError(f"Expected list, got {type(scores).__name__}")
            if len(scores) != len(queries):
                log.warning(
                    "ollama.score length mismatch",
                    expected=len(queries),
                    got=len(scores),
                )
                scores = (list(scores) + [0.5] * len(queries))[: len(queries)]
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"ollama score_relevance failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
