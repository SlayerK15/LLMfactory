"""Chunking stage: split each cleaned doc into fixed-size token windows with overlap."""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import structlog

from dataset_forge.core.models import Chunk, CleanedDoc, ForgeConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

log = structlog.get_logger()


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...


@lru_cache(maxsize=4)
def _load_hf_tokenizer(model_id: str) -> "PreTrainedTokenizerBase":
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def chunk_doc(doc: CleanedDoc, config: ForgeConfig, tokenizer: Tokenizer) -> list[Chunk]:
    """Split one doc into overlapping token windows. Last window may be shorter."""
    ids = tokenizer.encode(doc.text, add_special_tokens=False)
    window = config.chunk_tokens
    overlap = config.chunk_overlap_tokens
    stride = max(1, window - overlap)
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(ids), stride)):
        end = start + window
        slice_ids = ids[start:end]
        if not slice_ids:
            break
        text = tokenizer.decode(slice_ids, skip_special_tokens=True).strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                id=Chunk.make_id(doc.id, i),
                doc_id=doc.id,
                run_id=doc.run_id,
                url=doc.url,
                text=text,
                token_count=len(slice_ids),
                chunk_index=i,
            )
        )
        if end >= len(ids):
            break
    return chunks


def run(docs: list[CleanedDoc], config: ForgeConfig, tokenizer: Tokenizer | None = None) -> list[Chunk]:
    tok: Tokenizer = tokenizer if tokenizer is not None else _load_hf_tokenizer(config.tokenizer_model_id)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_doc(doc, config, tok))
    log.info("stage.chunk.done", input_docs=len(docs), output_chunks=len(all_chunks))
    return all_chunks
