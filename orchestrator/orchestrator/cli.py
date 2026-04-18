"""Typer CLI for local orchestrator runs."""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
import uvicorn

from orchestrator.runner import run_full_pipeline

app = typer.Typer(help="orchestrator — run or serve the end-to-end pipeline")


@app.command("serve")
def serve(host: str = "0.0.0.0", port: int = 8080, reload: bool = False) -> None:
    """Serve the FastAPI app + frontend."""
    uvicorn.run("orchestrator.app:app", host=host, port=port, reload=reload)


@app.command("run")
def run(
    topic: str = typer.Argument(...),
    doc_count: int = typer.Option(500),
    output_mode: str = typer.Option("pretrain"),
    data_dir: Path = typer.Option(Path("./data")),
) -> None:
    """Run the full pipeline from the command line, printing events as JSON lines."""

    async def _main() -> None:
        async for event in run_full_pipeline(
            topic=topic, doc_count=doc_count, output_mode=output_mode, data_dir=data_dir
        ):
            typer.echo(event.model_dump_json())

    asyncio.run(_main())


if __name__ == "__main__":
    app()
