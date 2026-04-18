"""Time-budget math for QLoRA training on an A100-class GPU.

Empirical throughput ballparks (tokens/sec/epoch under QLoRA on A100-40GB, batch 2×4,
rank 16, 4-bit base) — conservative:

    1 B  params  -> ~60k tok/s
    3 B  params  -> ~24k tok/s
    7 B  params  -> ~12k tok/s
    14 B params  -> ~5.5k tok/s

These decay roughly 1/params. Add ~3 min setup + ~3 min merge+GGUF+quant overhead.
"""
from __future__ import annotations

from dataclasses import dataclass

from trainer_service.core.errors import BudgetExceededError
from trainer_service.core.models import ModelSpec

# (params_b, default tok/s on A100-40GB QLoRA 4-bit). Interpolated for in-between sizes.
_BASELINES: list[tuple[float, float]] = [
    (1.0, 60_000.0),
    (3.0, 24_000.0),
    (7.0, 12_000.0),
    (14.0, 5_500.0),
]

SETUP_OVERHEAD_S = 180
POSTPROC_OVERHEAD_S = 180


def tokens_per_second(params_b: float) -> float:
    if params_b <= _BASELINES[0][0]:
        return _BASELINES[0][1]
    if params_b >= _BASELINES[-1][0]:
        return _BASELINES[-1][1]
    for (p0, t0), (p1, t1) in zip(_BASELINES, _BASELINES[1:], strict=False):
        if p0 <= params_b <= p1:
            frac = (params_b - p0) / (p1 - p0)
            return t0 + (t1 - t0) * frac
    return _BASELINES[-1][1]


@dataclass(frozen=True)
class BudgetEstimate:
    params_b: float
    train_tokens: int
    epochs: int
    estimated_s: int
    fits_budget: bool
    slack_s: int


def estimate(params_b: float, train_tokens: int, epochs: int, budget_s: int) -> BudgetEstimate:
    train_s = (train_tokens * epochs) / max(tokens_per_second(params_b), 1.0)
    total = int(train_s + SETUP_OVERHEAD_S + POSTPROC_OVERHEAD_S)
    slack = budget_s - total
    return BudgetEstimate(
        params_b=params_b,
        train_tokens=train_tokens,
        epochs=epochs,
        estimated_s=total,
        fits_budget=slack >= 0,
        slack_s=slack,
    )


def pick_largest_model_that_fits(
    candidates: list[ModelSpec], train_tokens: int, epochs: int, budget_s: int
) -> ModelSpec:
    """Among candidates, return the largest model whose estimate fits under budget_s."""
    sorted_desc = sorted(candidates, key=lambda m: m.params_b, reverse=True)
    for spec in sorted_desc:
        est = estimate(spec.params_b, train_tokens, epochs, budget_s)
        if est.fits_budget:
            return spec
    raise BudgetExceededError(
        f"No candidate fits budget: {budget_s}s (smallest={sorted_desc[-1].params_b}B)"
    )
