"""Q/A synthesis stage — wraps an LLM adapter to produce QAPairs from Chunks."""
from __future__ import annotations

from typing import Protocol

import structlog

from dataset_forge.core.models import Chunk, ForgeConfig, QAPair

log = structlog.get_logger()


class QASynthesiser(Protocol):
    async def synth(self, chunks: list[Chunk], max_concurrency: int) -> list[QAPair]: ...


async def run(chunks: list[Chunk], config: ForgeConfig, synth: QASynthesiser) -> list[QAPair]:
    pairs = await synth.synth(chunks, max_concurrency=config.qa_max_concurrency)
    log.info("stage.qa_synth.done", input_chunks=len(chunks), output_pairs=len(pairs))
    return pairs
