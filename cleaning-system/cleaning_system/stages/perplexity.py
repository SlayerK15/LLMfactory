"""
Perplexity-based quality filter.
Uses a small causal LM (Qwen2.5-0.5B) on CPU to compute per-document
cross-entropy loss. Drops statistical outliers in both tails:
  - bottom tail = extremely low perplexity → repetitive / templated content
  - top tail    = extremely high perplexity → incoherent / garbled content
"""
from __future__ import annotations

import structlog

from cleaning_system.core.errors import ModelLoadError
from cleaning_system.core.models import CleaningConfig, DocRecord

log = structlog.get_logger()


class _PerplexityModel:
    """Lazy-loaded wrapper around a CPU causal LM."""

    def __init__(self, model_id: str, max_tokens: int) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("perplexity.load_model", model_id=self._model_id)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            self._model.eval()
            log.info("perplexity.model_ready", model_id=self._model_id)
        except Exception as exc:
            raise ModelLoadError(self._model_id, str(exc)) from exc

    def compute(self, text: str) -> float:
        import torch

        self._load()
        inputs = self._tokenizer(
            text[:4096],
            return_tensors="pt",
            truncation=True,
            max_length=self._max_tokens,
        )
        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])
        return float(torch.exp(outputs.loss).item())


_cached_model: _PerplexityModel | None = None


def _get_model(config: CleaningConfig) -> _PerplexityModel:
    global _cached_model
    if _cached_model is None or _cached_model._model_id != config.perplexity_model_id:
        _cached_model = _PerplexityModel(
            config.perplexity_model_id, config.perplexity_max_tokens
        )
    return _cached_model


def run(docs: list[DocRecord], config: CleaningConfig) -> list[DocRecord]:
    """
    Score each document by perplexity and drop both-tail outliers.
    Skipped (returns docs unchanged) if torch is not installed.
    """
    if not docs:
        return docs

    try:
        import numpy as np
        import torch  # noqa: F401 — presence check
    except ImportError:
        log.warning("perplexity.skip", reason="torch or numpy not installed")
        return docs

    model = _get_model(config)

    log.info("perplexity.scoring", n_docs=len(docs))
    scores: list[float] = []
    for doc in docs:
        try:
            scores.append(model.compute(doc.text))
        except Exception as exc:
            log.warning("perplexity.score_error", doc_id=doc.id, error=str(exc))
            scores.append(float("nan"))

    valid = [s for s in scores if not (s != s)]  # filter NaN
    if len(valid) < 4:
        log.warning("perplexity.too_few_valid", n_valid=len(valid))
        return docs

    lo = float(np.nanpercentile(scores, config.perplexity_low_pct * 100))
    hi = float(np.nanpercentile(scores, (1.0 - config.perplexity_high_pct) * 100))

    kept = [doc for doc, s in zip(docs, scores) if lo <= s <= hi]

    log.info(
        "perplexity.done",
        input=len(docs),
        kept=len(kept),
        lo_threshold=round(lo, 2),
        hi_threshold=round(hi, 2),
    )
    return kept
