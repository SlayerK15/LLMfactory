"""Deterministic train/eval split keyed by doc_id — all chunks of one doc stay together."""
from __future__ import annotations

import hashlib
from typing import TypeVar

import structlog

from dataset_forge.core.models import Chunk, ForgeConfig, QAPair

T = TypeVar("T", Chunk, QAPair)

log = structlog.get_logger()


def _eval_bucket(doc_id: str, seed: int, eval_fraction: float) -> bool:
    h = hashlib.sha256(f"{seed}:{doc_id}".encode()).digest()
    bucket = int.from_bytes(h[:8], "big") / 2**64
    return bucket < eval_fraction


def split_by_doc(items: list[T], config: ForgeConfig) -> tuple[list[T], list[T]]:
    """Split items into (train, eval). All items sharing a doc_id go to the same split."""
    train: list[T] = []
    eval_: list[T] = []
    for item in items:
        if _eval_bucket(item.doc_id, config.split_seed, config.eval_fraction):
            eval_.append(item)
        else:
            train.append(item)
    log.info(
        "stage.split.done",
        total=len(items),
        train=len(train),
        eval=len(eval_),
        eval_fraction=config.eval_fraction,
    )
    return train, eval_
