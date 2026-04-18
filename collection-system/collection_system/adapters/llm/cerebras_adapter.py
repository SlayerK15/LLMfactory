"""Cerebras LLM adapter using the OpenAI-compatible API."""
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
from collection_system.core.errors import LLMError, LLMRateLimitError

log = structlog.get_logger()

DEFAULT_MODEL = "llama-3.3-70b"
DEFAULT_BASE_URL = "https://api.cerebras.ai"
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


class CerebrasAdapter:
    """Cloud LLM adapter for Cerebras-hosted models."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_depth: int = 3,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._max_depth = max_depth
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
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
            if resp.status_code == 429:
                raise LLMRateLimitError(f"Cerebras rate-limited request: {resp.text}")
            resp.raise_for_status()
            payload = resp.json()
        except LLMRateLimitError:
            raise
        except httpx.HTTPError as exc:
            raise LLMError(f"Cerebras request failed: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"Cerebras returned non-JSON: {exc}") from exc

        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Cerebras response malformed: {payload}") from exc

    async def expand_topic(self, topic: str, parent: str | None, depth: int) -> list[str]:
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
                "cerebras.expand_topic",
                expand_from=expand_from,
                depth=depth,
                count=len(result),
            )
            return result
        except (LLMRateLimitError, LLMError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"cerebras expand_topic failed: {exc}") from exc

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
                    "cerebras.score length mismatch",
                    expected=len(queries),
                    got=len(scores),
                )
                scores = (list(scores) + [0.5] * len(queries))[: len(queries)]
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except (LLMRateLimitError, LLMError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"cerebras score_relevance failed: {exc}") from exc

    async def health_check(self) -> bool:
        try:
            await self._chat("You are a health check assistant.", "Reply with: ok")
            return True
        except LLMError:
            return False
