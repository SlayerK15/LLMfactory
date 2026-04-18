"""CLI entry point for cleaning-system."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

app = typer.Typer(name="clean", help="Corpus cleaning pipeline for LLM fine-tuning.")


def _run(coro):
    return asyncio.run(coro)


@app.command()
def run(
    run_id: str = typer.Argument(..., help="Collection run ID to clean"),
    topic: str = typer.Option(..., "--topic", "-t", help="Original topic string"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir", help="Root data directory"),
    no_perplexity: bool = typer.Option(False, "--no-perplexity", help="Skip perplexity filter"),
    no_relevance: bool = typer.Option(False, "--no-relevance", help="Skip relevance filter"),
    no_trafilatura: bool = typer.Option(
        False, "--no-trafilatura", help="Skip Trafilatura re-extraction"
    ),
    relevance_threshold: float = typer.Option(
        0.30, "--relevance-threshold", help="Minimum cosine similarity to topic"
    ),
    near_dup_threshold: float = typer.Option(
        0.80, "--near-dup-threshold", help="MinHash Jaccard similarity threshold"
    ),
    language: str = typer.Option("en", "--language", help="Target language code (ISO 639-1)"),
):
    """Clean a collection run and produce a quality report."""
    from cleaning_system.api import run_cleaning
    from cleaning_system.infra.logging import configure_logging

    configure_logging()

    report = _run(
        run_cleaning(
            run_id=run_id,
            topic=topic,
            data_dir=data_dir,
            enable_perplexity=not no_perplexity,
            enable_relevance=not no_relevance,
            enable_trafilatura=not no_trafilatura,
            relevance_threshold=relevance_threshold,
            near_dup_threshold=near_dup_threshold,
            target_language=language,
        )
    )

    typer.echo("\n── Cleaning Report ─────────────────────────────")
    typer.echo(f"  Run ID          : {report.run_id}")
    typer.echo(f"  Topic           : {report.topic}")
    typer.echo(f"  Input docs      : {report.input_docs}")
    typer.echo(f"  After near-dedup: {report.after_near_dedup}")
    typer.echo(f"  After lang      : {report.after_lang_filter}")
    typer.echo(f"  After gopher    : {report.after_gopher}")
    typer.echo(f"  After perplexity: {report.after_perplexity}")
    typer.echo(f"  After relevance : {report.after_relevance}")
    typer.echo(f"  Total tokens    : {report.total_tokens:,}")
    typer.echo(f"  Median tokens   : {report.median_doc_tokens:,}")
    typer.echo(f"  Drop rate       : {report.drop_rate:.1%}")
    typer.echo(f"  Duration        : {report.cleaning_duration_s:.1f}s")
    typer.echo("────────────────────────────────────────────────\n")


@app.command()
def status(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
):
    """Show the cleaning report for a completed run."""
    report_path = data_dir / "reports" / f"{run_id}.json"
    if not report_path.exists():
        typer.echo(f"No report found at {report_path}", err=True)
        raise typer.Exit(1)

    data = json.loads(report_path.read_text())
    typer.echo(json.dumps(data, indent=2))


@app.command()
def list_runs(
    data_dir: Path = typer.Option(Path("./data"), "--data-dir"),
):
    """List all completed cleaning runs."""
    reports_dir = data_dir / "reports"
    if not reports_dir.exists():
        typer.echo("No cleaning reports found.")
        return

    reports = sorted(reports_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        typer.echo("No cleaning reports found.")
        return

    for path in reports:
        try:
            data = json.loads(path.read_text())
            typer.echo(
                f"{data['run_id']}  topic={data['topic']!r}  "
                f"output={data['after_relevance']}  "
                f"drop={data.get('drop_rate', 0):.1%}  "
                f"duration={data['cleaning_duration_s']:.0f}s"
            )
        except Exception:
            typer.echo(f"{path.stem}  (unreadable)")


if __name__ == "__main__":
    app()
