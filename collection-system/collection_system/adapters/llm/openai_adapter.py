"""OpenAI LLM adapter — pay-as-you-go alternative to Groq/Cerebras."""
from __future__ import annotations

import json
import re

import structlog
from openai import AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from collection_system.agents.prompts import (
    EXPAND_TOPIC_SYSTEM,
    EXPAND_TOPIC_USER,
    SCORE_RELEVANCE_SYSTEM,
    SCORE_RELEVANCE_USER,
    VALIDATE_URLS_SYSTEM,
    VALIDATE_URLS_USER,
)
from collection_system.core.errors import LLMError, LLMRateLimitError

log = structlog.get_logger()

DEFAULT_MODEL = "gpt-4o-mini"
EXPAND_COUNT = 12
SCORE_BATCH_SIZE = 20
VALIDATE_BATCH_SIZE = 25


def _parse_json_array(text: str) -> list:
    """Extract JSON array from LLM output, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


class OpenAIAdapter:
    """
    OpenAI API adapter for query expansion, relevance scoring, and URL
    validation. Pay-as-you-go — see docs for pricing.

    gpt-4o-mini is a good default for this workload (classification +
    short generation); swap to gpt-4o for higher quality at ~15× cost.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        max_depth: int = 3,
        base_url: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_depth = max_depth

    @retry(
        wait=wait_random_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitError),
        reraise=True,
    )
    async def _chat(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    async def expand_topic(
        self,
        topic: str,
        parent: str | None,
        depth: int,
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
            log.debug("openai.expand_topic", expand_from=expand_from, depth=depth, count=len(result))
            return result
        except RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"openai expand_topic failed: {exc}") from exc

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
                    "openai.score_relevance.length_mismatch",
                    expected=len(queries),
                    got=len(scores),
                )
                scores = (list(scores) + [0.5] * len(queries))[: len(queries)]
            return [max(0.0, min(1.0, float(s))) for s in scores]
        except RateLimitError as exc:
            raise LLMRateLimitError(str(exc)) from exc
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"openai score_relevance failed: {exc}") from exc

    async def validate_urls(
        self,
        topic: str,
        query: str,
        items: list[tuple[str, str | None, str | None]],
    ) -> list[bool]:
        if not items:
            return []
        out: list[bool] = []
        for start in range(0, len(items), VALIDATE_BATCH_SIZE):
            chunk = items[start : start + VALIDATE_BATCH_SIZE]
            rendered = "\n".join(
                f"{i + 1}. {url} — {(title or '(no title)').strip()[:140]} — "
                f"{(snippet or '').strip()[:200]}"
                for i, (url, title, snippet) in enumerate(chunk)
            )
            user_prompt = VALIDATE_URLS_USER.format(
                topic=topic, query=query, items=rendered
            )
            try:
                raw = await self._chat(VALIDATE_URLS_SYSTEM, user_prompt)
                parsed = _parse_json_array(raw)
                if not isinstance(parsed, list) or len(parsed) != len(chunk):
                    log.warning(
                        "openai.validate_urls.length_mismatch",
                        expected=len(chunk),
                        got=len(parsed) if isinstance(parsed, list) else "n/a",
                    )
                    out.extend([True] * len(chunk))
                    continue
                out.extend(bool(int(v)) for v in parsed)
            except Exception as exc:  # noqa: BLE001
                log.warning("openai.validate_urls.failed", error=str(exc))
                out.extend([True] * len(chunk))
        return out

    async def health_check(self) -> bool:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(resp.choices)
        except Exception:
            return False
