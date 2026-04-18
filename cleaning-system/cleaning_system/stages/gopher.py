"""
Gopher-style quality heuristics.
Based on: Rae et al. (2021) "Scaling Language Models: Methods, Analysis & Insights
from Training Gopher" — Section A.1 filtering rules.
"""
from __future__ import annotations

import structlog

from cleaning_system.core.models import CleaningConfig, DocRecord

log = structlog.get_logger()

# Common English stop words used for density check
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "if", "about", "who", "which", "me", "when", "make", "can",
        "like", "time", "no", "just", "him", "know", "take", "people",
        "into", "year", "your", "some", "could", "them", "than", "then",
        "now", "look", "only", "come", "its", "over", "also", "back",
        "after", "use", "two", "how", "our", "work", "first", "well",
        "way", "even", "new", "want", "because", "any", "these", "give",
        "day", "most", "us",
    }
)

_DROP_REASONS = (
    "length",
    "mean_word_len",
    "symbol_ratio",
    "ellipsis_ratio",
    "duplicate_line_ratio",
    "stopword_density",
)


def _check(doc: DocRecord, config: CleaningConfig) -> str | None:
    """Return a drop reason string or None if the document passes."""
    words = doc.text.split()
    n_words = len(words)

    if not (config.min_words <= n_words <= config.max_words):
        return "length"

    mean_wl = sum(len(w) for w in words) / n_words
    if not (config.mean_word_len_min <= mean_wl <= config.mean_word_len_max):
        return "mean_word_len"

    symbol_count = doc.text.count("#") + doc.text.count("...")
    if symbol_count / n_words > config.max_symbol_word_ratio:
        return "symbol_ratio"

    lines = doc.text.splitlines()
    non_empty = [l for l in lines if l.strip()]

    if non_empty:
        ellipsis_count = sum(1 for l in non_empty if l.rstrip().endswith("..."))
        if ellipsis_count / len(non_empty) > config.max_ellipsis_line_ratio:
            return "ellipsis_ratio"

        unique_lines = {l.strip() for l in non_empty}
        dup_ratio = (len(non_empty) - len(unique_lines)) / len(non_empty)
        if dup_ratio > config.max_duplicate_line_ratio:
            return "duplicate_line_ratio"

    stop_count = sum(1 for w in words if w.lower() in _STOP_WORDS)
    if stop_count / n_words < config.min_stopword_density:
        return "stopword_density"

    return None


def run(docs: list[DocRecord], config: CleaningConfig) -> list[DocRecord]:
    """Apply Gopher quality heuristics. Returns the surviving documents."""
    if not docs:
        return docs

    kept: list[DocRecord] = []
    drop_counts: dict[str, int] = {r: 0 for r in _DROP_REASONS}

    for doc in docs:
        reason = _check(doc, config)
        if reason:
            drop_counts[reason] += 1
            log.debug("gopher.drop", doc_id=doc.id, reason=reason)
        else:
            kept.append(doc)

    log.info(
        "gopher.done",
        input=len(docs),
        kept=len(kept),
        **{f"dropped_{k}": v for k, v in drop_counts.items() if v},
    )
    return kept
