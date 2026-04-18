"""
Collection System — agent-driven data collection pipeline.

Public API:
    run_collection()            blocking run
    run_collection_streaming()  async event iterator
    RunConfig                   run configuration model
    RunHandle                   returned by run_collection()
"""
from collection_system.api import RunHandle, run_collection, run_collection_streaming
from collection_system.core.models import RunConfig

__all__ = [
    "run_collection",
    "run_collection_streaming",
    "RunConfig",
    "RunHandle",
]
