"""
CLI entry point.
Usage: collect <command> [options]
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from collection_system.core.constants import SearchBackend
from collection_system.core.models import RunConfig
from collection_system.infra.logging import configure_logging

app = typer.Typer(name="collect", help="Collection System CLI", no_args_is_help=True)


def _parse_backends(raw: str) -> list[SearchBackend]:
    mapping = {
        "cc_cdx": SearchBackend.CC_CDX,
        "searxng": SearchBackend.SEARXNG,
        "ddg_lite": SearchBackend.DDG_LITE,
    }
    out: list[SearchBackend] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise typer.BadParameter(f"Unknown backend: {key}")
        out.append(mapping[key])
    return out or [SearchBackend.CC_CDX, SearchBackend.SEARXNG]


# ---------------------------------------------------------------------------

@app.command()
def run(
    topic: str = typer.Argument(..., help="Topic to collect data about"),
    doc_count: int = typer.Option(2000, "--doc-count", "-n", help="Target document count"),
    config_file: Path = typer.Option(
        Path("configs/default.toml"), "--config", "-c"
    ),
    concurrency: int = typer.Option(40, "--concurrency"),
    backends: str = typer.Option(
        "cc_cdx,searxng",
        "--backends",
        help="Comma-separated: cc_cdx,searxng,ddg_lite",
    ),
    max_depth: int = typer.Option(3, "--max-depth"),
    max_queries: int = typer.Option(600, "--max-queries"),
) -> None:
    """Start a new collection run."""
    configure_logging()
    config = RunConfig(
        topic=topic,
        doc_count=doc_count,
        max_depth=max_depth,
        max_queries=max_queries,
        search_backends=_parse_backends(backends),
        scraper_concurrency=concurrency,
    )

    typer.secho(f"Starting run {config.run_id} for topic: {topic!r}", fg=typer.colors.CYAN)

    async def _go() -> None:
        from collection_system.api import run_collection_streaming

        async for event in run_collection_streaming(config):
            kind = getattr(event, "kind", type(event).__name__)
            typer.echo(f"[{kind}] {_event_summary(event)}")

    asyncio.run(_go())
    typer.secho(f"Run {config.run_id} finished.", fg=typer.colors.GREEN)


def _event_summary(event: object) -> str:
    """Short human-readable line per event."""
    kind = getattr(event, "kind", type(event).__name__)
    if kind == "QueriesGenerated":
        return f"count={getattr(event, 'count', '?')}"
    if kind == "URLsDiscovered":
        return (
            f"backend={getattr(event, 'backend', '?')} "
            f"count={getattr(event, 'count', '?')}"
        )
    if kind == "DocScraped":
        return (
            f"{getattr(event, 'url', '')} "
            f"tokens={getattr(event, 'token_count', 0)}"
        )
    if kind == "DocFailed":
        return (
            f"{getattr(event, 'url', '')} "
            f"err={getattr(event, 'error_type', '?')}"
        )
    if kind == "StageCompleted":
        stats = getattr(event, "stats", None)
        if stats:
            return (
                f"stage={getattr(stats, 'stage', '?').value if hasattr(getattr(stats, 'stage', None), 'value') else '?'} "
                f"out={getattr(stats, 'output_count', 0)} "
                f"fail={getattr(stats, 'failure_count', 0)}"
            )
        return ""
    if kind == "RunCompleted":
        return (
            f"docs={getattr(event, 'docs_collected', 0)} "
            f"dur={getattr(event, 'duration_s', 0):.1f}s"
        )
    if kind == "RunFailed":
        return f"error={getattr(event, 'error', '')}"
    return ""


@app.command()
def resume(run_id: str = typer.Argument(..., help="Run ID to resume")) -> None:
    """Resume a collection run from its last checkpoint. (TODO: Phase 1.1)"""
    typer.secho(
        f"Resume for {run_id} is not yet wired up — this is Phase 1.1 work.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=2)


@app.command()
def status(run_id: str = typer.Argument(...)) -> None:
    """Print stage-by-stage progress for a run."""
    configure_logging()

    async def _go() -> None:
        from collection_system.bootstrap import build_adapters

        cfg = RunConfig(run_id=run_id, topic="")
        adapters = await build_adapters(cfg)
        manifest = await adapters.storage.load_run(run_id)
        typer.secho(f"Run {run_id} — {manifest.status.value}", fg=typer.colors.CYAN)
        typer.echo(f"Topic: {manifest.config.topic}")
        typer.echo(f"Started: {manifest.started_at.isoformat()}")
        if manifest.completed_at:
            typer.echo(f"Completed: {manifest.completed_at.isoformat()}")
        typer.echo("")
        typer.echo(f"{'Stage':<20} {'Status':<12} {'In':>6} {'Out':>6} {'Fail':>6} {'Dur(s)':>8}")
        for stage, stats in manifest.stages.items():
            done = "done" if stats.completed_at else "running"
            dur = f"{stats.duration_s:.1f}" if stats.duration_s else "-"
            typer.echo(
                f"{stage.value:<20} {done:<12} "
                f"{stats.input_count:>6} {stats.output_count:>6} "
                f"{stats.failure_count:>6} {dur:>8}"
            )

    asyncio.run(_go())


@app.command("list")
def list_runs(limit: int = typer.Option(20, "--limit", "-l")) -> None:
    """List recent runs with status, doc counts, and timing."""
    configure_logging()

    async def _go() -> None:
        from collection_system.bootstrap import build_adapters

        cfg = RunConfig(run_id="list-probe", topic="")
        adapters = await build_adapters(cfg)
        rows = await adapters.storage.list_runs(limit=limit)
        typer.echo(f"{'Run ID':<40} {'Topic':<30} {'Status':<10} {'Docs':>6} {'Dur(s)':>8}")
        for r in rows:
            dur = f"{r.duration_s:.1f}" if r.duration_s else "-"
            typer.echo(
                f"{r.run_id:<40} {r.topic[:28]:<30} "
                f"{r.status.value:<10} {r.docs_collected:>6} {dur:>8}"
            )

    asyncio.run(_go())


@app.command()
def failures(run_id: str = typer.Argument(...)) -> None:
    """Print failure log for a run."""
    configure_logging()

    async def _go() -> None:
        from collection_system.bootstrap import build_adapters

        cfg = RunConfig(run_id=run_id, topic="")
        adapters = await build_adapters(cfg)
        fails = await adapters.storage.load_failures(run_id)
        typer.secho(f"{len(fails)} failures for run {run_id}", fg=typer.colors.YELLOW)
        for f in fails:
            typer.echo(
                f"[{f.stage.value}] {f.error_type}: {f.target} — {f.error_msg}"
            )

    asyncio.run(_go())


@app.command()
def inspect(run_id: str = typer.Argument(...)) -> None:
    """Print full manifest for a run."""
    configure_logging()

    async def _go() -> None:
        from collection_system.bootstrap import build_adapters

        cfg = RunConfig(run_id=run_id, topic="")
        adapters = await build_adapters(cfg)
        manifest = await adapters.storage.load_run(run_id)
        typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2))

    asyncio.run(_go())


@app.command()
def health() -> None:
    """Ping all adapters (LLM, search backends, database)."""
    configure_logging()

    async def _go() -> None:
        from collection_system.bootstrap import build_adapters

        cfg = RunConfig(run_id="health-probe", topic="health")
        adapters = await build_adapters(cfg)

        results: dict[str, bool] = {}
        results["llm"] = await adapters.llm.health_check()
        results["search"] = await adapters.search.health_check()
        try:
            results["db"] = await adapters.storage.run_exists(
                "00000000-0000-0000-0000-000000000000"
            ) or True
        except Exception:  # noqa: BLE001
            results["db"] = False

        for name, ok in results.items():
            color = typer.colors.GREEN if ok else typer.colors.RED
            marker = "OK" if ok else "FAIL"
            typer.secho(f"{name:<10} {marker}", fg=color)

    asyncio.run(_go())


@app.command()
def db_migrate() -> None:
    """Run Alembic migrations (upgrade head)."""
    import subprocess
    subprocess.run(["uv", "run", "alembic", "upgrade", "head"], check=True)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Start the FastAPI server (for future orchestrator integration)."""
    import uvicorn

    uvicorn.run("collection_system.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
