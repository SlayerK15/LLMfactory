"""Smoke test: can we import the Modal app and inspect its structure without dispatching?

The real end-to-end run is prohibitively expensive to put in CI (A100 time). This test
verifies the Modal app wiring is correct — catches typos in decorators, image build spec,
function signatures — without actually launching GPU work.
"""
from __future__ import annotations

import pytest


@pytest.mark.integration
def test_modal_app_imports():
    modal = pytest.importorskip("modal")
    from trainer_service.training.modal_app import APP_NAME, app, train

    assert APP_NAME == "collection-system-trainer"
    assert isinstance(app, modal.App)
    assert callable(train)
