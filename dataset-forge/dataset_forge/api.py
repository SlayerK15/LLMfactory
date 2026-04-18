"""Public async API — matches the pattern used by collection-system and cleaning-system."""
from __future__ import annotations

from dataset_forge.core.models import ForgeConfig, ForgeReport
from dataset_forge.core.pipeline import run_forge as _run_forge


async def run_forge(config: ForgeConfig) -> ForgeReport:
    """Build a training dataset from a cleaned run. See ForgeConfig for knobs."""
    return await _run_forge(config)
