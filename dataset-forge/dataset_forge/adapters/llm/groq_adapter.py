"""Groq-backed Q/A synthesiser — returns JSON list of {instruction, response} per chunk."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from dataset_forge.core.errors import QASynthError
from dataset_forge.core.models import Chunk, ForgeConfig, QAPair

log = structlog.get_logger()


_SYSTEM_PROMPT = (
    "You are a dataset curator creating instruction-tuning examples from source text. "
    "For each passage, generate exactly N diverse question-answer pairs that a learner "
    "would find useful. Questions must be answerable from the passage alone. "
    "Respond ONLY with a JSON array of objects, each with keys 'instruction' and 'response'. "
    "Keep responses concise (2-5 sentences) and grounded in the passage."
)


def _build_user_message(chunk: Chunk, n_pairs: int) -> str:
    return (
        f"Generate exactly {n_pairs} instruction-response pairs grounded in this passage.\n"
        f"Return a JSON array of {n_pairs} objects with keys 'instruction' and 'response'.\n\n"
        f"PASSAGE:\n{chunk.text}"
    )


class GroqQASynth:
    def __init__(self, config: ForgeConfig, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise QASynthError("GROQ_API_KEY not set")
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=key, timeout=config.qa_timeout_s)
        self._model = config.qa_model
        self._n = config.qa_pairs_per_chunk
        self._min_instr = config.qa_min_instruction_len
        self._min_resp = config.qa_min_response_len

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=20.0),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_once(self, chunk: Chunk) -> list[dict[str, Any]]:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT.replace("N", str(self._n))},
                {"role": "user", "content": _build_user_message(chunk, self._n)},
            ],
            response_format={"type": "json_object"},
            temperature=0.4,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)
        if isinstance(data, dict):
            for key in ("pairs", "items", "data", "qa"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                data = list(data.values())[0] if data else []
        if not isinstance(data, list):
            raise QASynthError(f"unexpected QA response shape: {type(data).__name__}")
        return data

    async def synth_for_chunk(self, chunk: Chunk) -> list[QAPair]:
        try:
            raw = await self._call_once(chunk)
        except Exception as exc:
            log.warning("qa_synth.chunk_failed", chunk_id=chunk.id, error=str(exc))
            return []
        pairs: list[QAPair] = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            instr = str(item.get("instruction", "")).strip()
            resp = str(item.get("response", item.get("answer", ""))).strip()
            if len(instr) < self._min_instr or len(resp) < self._min_resp:
                continue
            pairs.append(
                QAPair(
                    id=QAPair.make_id(chunk.id, i),
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    run_id=chunk.run_id,
                    instruction=instr,
                    response=resp,
                )
            )
        return pairs

    async def synth(self, chunks: list[Chunk], max_concurrency: int) -> list[QAPair]:
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(c: Chunk) -> list[QAPair]:
            async with sem:
                return await self.synth_for_chunk(c)

        results = await asyncio.gather(*[_one(c) for c in chunks])
        return [qa for batch in results for qa in batch]
