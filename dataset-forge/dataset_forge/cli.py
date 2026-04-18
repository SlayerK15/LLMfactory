"""Typer CLI for dataset-forge."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from dataset_forge.core.models import ForgeConfig, OutputMode
from dataset_forge.core.pipeline import run_forge
from dataset_forge.infra.logging import configure

app = typer.Typer(help="dataset-forge — cleaned corpus → training-ready JSONL")


@app.command("build")
def build(
    run_id: str = typer.Argument(..., help="Run id produced by collection-system / cleaning-system"),
    topic: str = typer.Argument(..., help="Topic string (carried into the dataset card)"),
    mode: str = typer.Option("pretrain", help="pretrain | instruct"),
    data_dir: Path = typer.Option(Path("./data"), help="Root data dir"),
    chunk_tokens: int = typer.Option(2048),
    chunk_overlap: int = typer.Option(200),
    qa_pairs_per_chunk: int = typer.Option(3),
    qa_concurrency: int = typer.Option(4),
    eval_fraction: float = typer.Option(0.10),
    no_embed: bool = typer.Option(False, "--no-embed", help="Skip BGE-M3 → LanceDB"),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure(log_level)
    config = ForgeConfig(
        run_id=run_id,
        topic=topic,
        data_dir=data_dir,
        output_mode=OutputMode(mode),
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=chunk_overlap,
        qa_pairs_per_chunk=qa_pairs_per_chunk,
        qa_max_concurrency=qa_concurrency,
        eval_fraction=eval_fraction,
        enable_embeddings=not no_embed,
    )
    report = asyncio.run(run_forge(config))
    typer.echo(json.dumps(report.model_dump(), indent=2, default=str))


@app.command("inspect")
def inspect(
    run_id: str = typer.Argument(...),
    data_dir: Path = typer.Option(Path("./data")),
) -> None:
    card_path = data_dir / "datasets" / f"{run_id}.card.json"
    if not card_path.exists():
        typer.echo(f"no card found at {card_path}", err=True)
        raise typer.Exit(1)
    typer.echo(card_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    app()
