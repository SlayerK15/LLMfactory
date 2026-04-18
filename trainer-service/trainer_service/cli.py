"""Typer CLI for trainer-service."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from trainer_service.core.models import OutputFormat, TrainConfig, TrainingStyle
from trainer_service.infra.budget import estimate
from trainer_service.infra.logging import configure

app = typer.Typer(help="trainer-service — Modal QLoRA fine-tuner")


@app.command("plan")
def plan(
    params_b: float = typer.Option(3.0, help="Base model size in billions of parameters"),
    train_tokens: int = typer.Option(..., help="Total training tokens (from dataset card)"),
    epochs: int = typer.Option(1),
    budget_s: int = typer.Option(30 * 60, help="Time budget in seconds"),
) -> None:
    est = estimate(params_b, train_tokens, epochs, budget_s)
    typer.echo(
        json.dumps(
            {
                "params_b": est.params_b,
                "estimated_s": est.estimated_s,
                "fits_budget": est.fits_budget,
                "slack_s": est.slack_s,
            },
            indent=2,
        )
    )


@app.command("train")
def train(
    run_id: str = typer.Argument(...),
    topic: str = typer.Argument(...),
    train_jsonl: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    eval_jsonl: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    base_model: str = typer.Option("unsloth/Llama-3.2-3B-Instruct-bnb-4bit"),
    params_b: float = typer.Option(3.0),
    style: str = typer.Option("instruct", help="pretrain | instruct"),
    epochs: int = typer.Option(1),
    lora_rank: int = typer.Option(16),
    output_format: str = typer.Option("gguf_q4_k_m"),
    s3_bucket: str = typer.Option("", help="If set, uploads the artifact to s3://<bucket>/..."),
    workspace: Path = typer.Option(Path("./data/trainer")),
    log_level: str = typer.Option("INFO"),
) -> None:
    configure(log_level)
    from trainer_service.api import run_training

    config = TrainConfig(
        run_id=run_id,
        topic=topic,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        base_model=base_model,
        params_b=params_b,
        training_style=TrainingStyle(style),
        epochs=epochs,
        lora_rank=lora_rank,
        output_format=OutputFormat(output_format),
        s3_bucket=s3_bucket or None,
        workspace=workspace,
    )
    report = asyncio.run(run_training(config))
    typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    app()
