"""Near-duplicate removal using MinHash LSH (datasketch)."""
from __future__ import annotations

import structlog
from datasketch import MinHash, MinHashLSH

from cleaning_system.core.models import CleaningConfig, DocRecord

log = structlog.get_logger()


def _shingles(text: str, k: int = 5) -> set[bytes]:
    words = text.lower().split()
    if len(words) < k:
        return {text.lower().encode("utf-8")}
    return {
        " ".join(words[i : i + k]).encode("utf-8")
        for i in range(len(words) - k + 1)
    }


def run(docs: list[DocRecord], config: CleaningConfig) -> list[DocRecord]:
    """
    Remove near-duplicate documents using MinHash LSH.
    Documents with Jaccard similarity >= threshold are considered duplicates;
    only the first-seen copy is kept.
    """
    if not docs:
        return docs

    lsh = MinHashLSH(
        threshold=config.near_dup_threshold,
        num_perm=config.near_dup_num_perm,
    )
    kept: list[DocRecord] = []
    dropped = 0

    for doc in docs:
        m = MinHash(num_perm=config.near_dup_num_perm)
        for shingle in _shingles(doc.text, config.near_dup_shingle_size):
            m.update(shingle)

        duplicates = lsh.query(m)
        if duplicates:
            dropped += 1
            log.debug(
                "near_dedup.drop",
                doc_id=doc.id,
                near_dup_of=duplicates[0],
            )
            continue

        lsh.insert(doc.id, m)
        kept.append(doc)

    log.info("near_dedup.done", input=len(docs), kept=len(kept), dropped=dropped)
    return kept
