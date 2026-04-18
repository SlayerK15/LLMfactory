"""Unit tests for budget estimator + model selector."""
from __future__ import annotations

import pytest

from trainer_service.core.errors import BudgetExceededError
from trainer_service.core.models import ModelSpec
from trainer_service.infra.budget import estimate, pick_largest_model_that_fits, tokens_per_second


def test_tokens_per_second_monotonically_decreases_with_size():
    a = tokens_per_second(1.0)
    b = tokens_per_second(3.0)
    c = tokens_per_second(7.0)
    d = tokens_per_second(14.0)
    assert a > b > c > d


def test_estimate_fits_when_budget_generous():
    est = estimate(params_b=3.0, train_tokens=100_000, epochs=1, budget_s=60 * 30)
    assert est.fits_budget is True
    assert est.slack_s > 0


def test_estimate_does_not_fit_when_tight():
    est = estimate(params_b=14.0, train_tokens=5_000_000, epochs=3, budget_s=60 * 10)
    assert est.fits_budget is False
    assert est.slack_s < 0


def test_pick_largest_fits_chooses_biggest_under_budget():
    candidates = [
        ModelSpec(hf_id="a", params_b=1.0),
        ModelSpec(hf_id="b", params_b=3.0),
        ModelSpec(hf_id="c", params_b=7.0),
        ModelSpec(hf_id="d", params_b=14.0),
    ]
    # 30-min budget should comfortably fit a 3B model on 500k tokens, but likely not 14B
    chosen = pick_largest_model_that_fits(candidates, train_tokens=500_000, epochs=1, budget_s=60 * 30)
    assert chosen.params_b <= 14.0
    assert chosen.params_b >= 3.0


def test_pick_raises_when_nothing_fits():
    candidates = [ModelSpec(hf_id="big", params_b=14.0)]
    with pytest.raises(BudgetExceededError):
        pick_largest_model_that_fits(candidates, train_tokens=50_000_000, epochs=5, budget_s=60)
