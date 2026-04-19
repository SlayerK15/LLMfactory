"""Tests for the FastAPI app — health, index, dry-run SSE."""
from __future__ import annotations

import json
import logging

import anyio
import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from orchestrator.app import _pump_with_logs, create_app
from orchestrator.events import OrchestratorEvent, Stage


@pytest.fixture(autouse=True)
def _reset_sse_app_status():
    """sse-starlette stores should_exit_event at module scope, which binds to the first
    event loop that touches it and then explodes on subsequent TestClient runs. Rebind
    it per test so every streaming case gets a fresh event on the right loop."""
    AppStatus.should_exit_event = anyio.Event()
    yield
    AppStatus.should_exit_event = anyio.Event()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_html(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Collection System" in r.text


def test_dry_run_streams_events(client: TestClient):
    with client.stream("POST", "/api/pipeline/dry-run", json={"topic": "DevOps", "doc_count": 42}) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        raw = "".join(chunk for chunk in r.iter_text())

    events = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append(json.loads(payload))

    kinds = [(e["kind"], e["stage"]) for e in events]
    assert ("stage_started", "init") == kinds[0]
    assert ("pipeline_done", "done") == kinds[-1]
    assert any(k == ("stage_done", "collect") for k in kinds)
    assert any(k == ("stage_done", "forge") for k in kinds)


def test_invalid_topic_rejected(client: TestClient):
    r = client.post("/api/pipeline/dry-run", json={"topic": "", "doc_count": 500})
    assert r.status_code == 422


def test_dry_run_train_mode_streams_train_stage(client: TestClient):
    """Dry-run with mode='train' must emit a train stage."""
    body = {
        "topic": "DevOps",
        "doc_count": 50,
        "mode": "train",
        "epochs": 2,
        "lora_rank": 8,
    }
    with client.stream("POST", "/api/pipeline/dry-run", json=body) as r:
        assert r.status_code == 200
        raw = "".join(chunk for chunk in r.iter_text())

    events = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append(json.loads(payload))

    kinds = [(e["kind"], e["stage"]) for e in events]
    assert ("stage_started", "train") in kinds
    assert ("stage_done", "train") in kinds
    assert kinds[-1] == ("pipeline_done", "done")

    # pipeline_done payload carries the train fake result with our knobs.
    final = events[-1]
    train_payload = final["data"]["train"]
    assert train_payload["trained"] is True
    assert train_payload["epochs"] == 2


def test_dataset_only_mode_does_not_emit_train_stage(client: TestClient):
    body = {"topic": "DevOps", "doc_count": 50, "mode": "dataset_only"}
    with client.stream("POST", "/api/pipeline/dry-run", json=body) as r:
        assert r.status_code == 200
        raw = "".join(chunk for chunk in r.iter_text())

    events = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append(json.loads(payload))

    stages_seen = {e["stage"] for e in events}
    assert "train" not in stages_seen


def test_invalid_mode_rejected(client: TestClient):
    r = client.post(
        "/api/pipeline/dry-run",
        json={"topic": "DevOps", "doc_count": 50, "mode": "fine-tune-plz"},
    )
    assert r.status_code == 422


def test_invalid_output_format_rejected(client: TestClient):
    r = client.post(
        "/api/pipeline/dry-run",
        json={
            "topic": "DevOps", "doc_count": 50,
            "mode": "train", "output_format": "onnx",
        },
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_pump_with_logs_includes_crawl4ai_stdlib_logs():
    async def fake_pipeline():
        logging.getLogger("crawl4ai.browser").info("launched browser")
        yield OrchestratorEvent(
            kind="stage_started",
            stage=Stage.INIT,
            run_id="test-run",
            message="booting",
        )
        yield OrchestratorEvent(
            kind="pipeline_done",
            stage=Stage.DONE,
            run_id="test-run",
            message="done",
            data={},
        )

    items = []
    async for item in _pump_with_logs(fake_pipeline()):
        items.append(item)

    log_payloads = [
        json.loads(item["data"])
        for item in items
        if item["event"] == "log"
    ]
    assert any(
        "crawl4ai.browser" in payload["message"]
        and "launched browser" in payload["message"]
        for payload in log_payloads
    )
