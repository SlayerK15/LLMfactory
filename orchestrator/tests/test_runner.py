"""Unit tests for runner.run_full_pipeline — uses fake stage functions."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.events import Stage
from orchestrator.runner import run_full_pipeline


async def _collect_ok(*, topic: str, run_id: str, data_dir: Path, doc_count: int) -> dict:
    return {"topic": topic, "doc_count": doc_count}


async def _clean_ok(*, topic: str, run_id: str, data_dir: Path) -> dict:
    return {"kept": 42}


async def _forge_ok(*, topic: str, run_id: str, data_dir: Path, output_mode: str) -> dict:
    return {"mode": output_mode}


async def test_runner_emits_expected_event_sequence(tmp_path: Path):
    events = []
    async for evt in run_full_pipeline(
        topic="DevOps",
        data_dir=tmp_path,
        collect=_collect_ok,
        clean=_clean_ok,
        forge=_forge_ok,
    ):
        events.append(evt)

    kinds = [(e.kind, e.stage) for e in events]
    assert kinds[0] == ("stage_started", Stage.INIT)
    assert ("stage_started", Stage.COLLECT) in kinds
    assert ("stage_done", Stage.COLLECT) in kinds
    assert ("stage_started", Stage.CLEAN) in kinds
    assert ("stage_done", Stage.CLEAN) in kinds
    assert ("stage_started", Stage.FORGE) in kinds
    assert ("stage_done", Stage.FORGE) in kinds
    assert kinds[-1] == ("pipeline_done", Stage.DONE)


async def test_runner_stops_on_first_failure(tmp_path: Path):
    async def boom(**kw):
        raise RuntimeError("nope")

    events = []
    async for evt in run_full_pipeline(
        topic="DevOps",
        data_dir=tmp_path,
        collect=_collect_ok,
        clean=boom,
        forge=_forge_ok,
    ):
        events.append(evt)

    # Last event must be the error; forge must not have started
    assert events[-1].kind == "error"
    assert events[-1].stage == Stage.FAILED
    stages_started = [e.stage for e in events if e.kind == "stage_started"]
    assert Stage.FORGE not in stages_started


async def test_runner_accepts_custom_run_id(tmp_path: Path):
    events = []
    async for evt in run_full_pipeline(
        topic="x", run_id="fixed-id", data_dir=tmp_path,
        collect=_collect_ok, clean=_clean_ok, forge=_forge_ok,
    ):
        events.append(evt)
    assert all(e.run_id == "fixed-id" for e in events)


async def test_default_mode_skips_train_stage(tmp_path: Path):
    """When mode is omitted (default 'dataset_only'), train must never be invoked."""
    train_calls: list[dict] = []

    async def _train_should_not_run(**kwargs):  # pragma: no cover — must stay unreached
        train_calls.append(kwargs)
        return {"trained": True}

    stages_started = []
    async for evt in run_full_pipeline(
        topic="x",
        data_dir=tmp_path,
        collect=_collect_ok,
        clean=_clean_ok,
        forge=_forge_ok,
        train=_train_should_not_run,
    ):
        if evt.kind == "stage_started":
            stages_started.append(evt.stage)

    assert Stage.TRAIN not in stages_started
    assert train_calls == []


async def test_train_mode_appends_train_stage(tmp_path: Path):
    """mode='train' must run train after forge and include it in results."""
    train_seen: dict = {}

    async def _train_ok(
        *,
        topic: str,
        run_id: str,
        data_dir: Path,
        output_mode: str,
        train_config: dict,
    ) -> dict:
        train_seen.update(
            topic=topic, run_id=run_id, data_dir=data_dir,
            output_mode=output_mode, train_config=train_config,
        )
        return {"trained": True, "epochs": train_config.get("epochs")}

    events = []
    async for evt in run_full_pipeline(
        topic="Kubernetes",
        data_dir=tmp_path,
        mode="train",
        train_config={"epochs": 3, "lora_rank": 8},
        collect=_collect_ok,
        clean=_clean_ok,
        forge=_forge_ok,
        train=_train_ok,
    ):
        events.append(evt)

    kinds = [(e.kind, e.stage) for e in events]
    # Train must appear after forge, before pipeline_done.
    assert ("stage_started", Stage.TRAIN) in kinds
    assert ("stage_done", Stage.TRAIN) in kinds
    forge_done_idx = kinds.index(("stage_done", Stage.FORGE))
    train_started_idx = kinds.index(("stage_started", Stage.TRAIN))
    assert forge_done_idx < train_started_idx
    assert kinds[-1] == ("pipeline_done", Stage.DONE)

    # Train was called with the right knobs.
    assert train_seen["topic"] == "Kubernetes"
    assert train_seen["output_mode"] == "pretrain"
    assert train_seen["train_config"] == {"epochs": 3, "lora_rank": 8}

    # Final pipeline_done payload must carry the train result.
    final = events[-1]
    assert final.data["train"] == {"trained": True, "epochs": 3}


async def test_train_mode_failure_halts_pipeline(tmp_path: Path):
    """A failing train stage must surface an error and not yield pipeline_done."""
    async def _train_boom(**kwargs):
        raise RuntimeError("gpu on fire")

    events = []
    async for evt in run_full_pipeline(
        topic="x",
        data_dir=tmp_path,
        mode="train",
        collect=_collect_ok,
        clean=_clean_ok,
        forge=_forge_ok,
        train=_train_boom,
    ):
        events.append(evt)

    assert events[-1].kind == "error"
    assert "gpu on fire" in events[-1].message
    assert not any(e.kind == "pipeline_done" for e in events)
