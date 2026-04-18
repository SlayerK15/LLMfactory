"""In-container training logic — called by Modal or a local GPU runner.

Pure module: does not depend on Modal. Takes a config dict + raw JSONL bytes, returns a dict
with artifact bytes and metrics. Heavy deps are imported lazily so this module can be inspected
(and partially unit-tested) on machines without Unsloth / CUDA.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _materialise_jsonl(raw: bytes, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sum(1 for _ in path.read_text(encoding="utf-8").splitlines() if _.strip())


def _count_tokens_approx(raw: bytes) -> int:
    # Rough token count for reporting (no tokenizer dep in this helper).
    return int(len(raw.decode("utf-8", errors="ignore").split()) * 1.3)


def _format_record(rec: dict[str, Any], style: str) -> str:
    if style == "pretrain":
        return rec.get("text", "")
    instr = rec.get("instruction", "")
    inp = rec.get("input", "")
    resp = rec.get("response", rec.get("answer", ""))
    prompt = f"### Instruction:\n{instr}\n\n"
    if inp:
        prompt += f"### Input:\n{inp}\n\n"
    prompt += f"### Response:\n{resp}"
    return prompt


def train_and_export(
    config: dict[str, Any],
    train_bytes: bytes,
    eval_bytes: bytes,
    cache_dir: str = "/cache",
) -> dict[str, Any]:
    """Run QLoRA, merge, convert to GGUF (if requested), return bytes + metrics.

    This function is executed inside the Modal container (CUDA + Unsloth available).
    Do not call locally unless you have the same environment.
    """
    from datasets import Dataset  # type: ignore[import-not-found]
    from trl import SFTConfig, SFTTrainer  # type: ignore[import-not-found]
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    run_id = config["run_id"]
    workdir = Path(cache_dir) / "runs" / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    train_path = workdir / "train.jsonl"
    eval_path = workdir / "eval.jsonl"
    n_train = _materialise_jsonl(train_bytes, train_path)
    n_eval = _materialise_jsonl(eval_bytes, eval_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["base_model"],
        max_seq_length=config["context_window"],
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
        random_state=config["seed"],
    )

    style = config["training_style"]

    def _to_dataset(path: Path) -> "Dataset":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [_format_record(r, style) for r in records]
        return Dataset.from_dict({"text": texts})

    train_ds = _to_dataset(train_path)
    eval_ds = _to_dataset(eval_path) if n_eval else None

    sft_args = SFTConfig(
        output_dir=str(workdir / "out"),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["grad_accum_steps"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=25,
        save_strategy="no",
        seed=config["seed"],
        bf16=True,
        dataset_text_field="text",
        max_seq_length=config["context_window"],
    )

    t0 = time.monotonic()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    train_metrics = trainer.train()
    train_loss = float(getattr(train_metrics, "training_loss", 0.0))
    eval_metrics = trainer.evaluate() if eval_ds is not None else {}
    eval_loss = float(eval_metrics.get("eval_loss", 0.0)) if eval_metrics else None
    train_duration = time.monotonic() - t0

    merged_dir = workdir / "merged"
    merged_dir.mkdir(exist_ok=True)
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    output_format = config["output_format"]
    artifact_path = workdir / "artifact.bin"
    if output_format.startswith("gguf"):
        quant = "q4_k_m" if output_format == "gguf_q4_k_m" else "q8_0"
        _convert_to_gguf(merged_dir, artifact_path, quant)
    else:
        import shutil

        shutil.make_archive(str(artifact_path), "zip", str(merged_dir))
        artifact_path = artifact_path.with_suffix(".bin.zip")

    artifact_bytes = artifact_path.read_bytes()
    sha = hashlib.sha256(artifact_bytes).hexdigest()

    return {
        "artifact_bytes": artifact_bytes,
        "artifact_sha256": sha,
        "metrics": {
            "train_loss_final": train_loss,
            "eval_loss_final": eval_loss,
            "train_tokens": _count_tokens_approx(train_bytes),
            "eval_tokens": _count_tokens_approx(eval_bytes),
            "n_train_records": n_train,
            "n_eval_records": n_eval,
            "train_duration_s": train_duration,
        },
    }


def _convert_to_gguf(merged_dir: Path, out_path: Path, quant: str) -> None:
    """Use the bundled llama.cpp tooling to produce an int4 GGUF."""
    import subprocess

    fp16_path = out_path.with_suffix(".fp16.gguf")
    subprocess.run(
        [
            "python",
            "/opt/llama.cpp/convert_hf_to_gguf.py",
            str(merged_dir),
            "--outtype",
            "f16",
            "--outfile",
            str(fp16_path),
        ],
        check=True,
    )
    subprocess.run(
        ["/opt/llama.cpp/build/bin/llama-quantize", str(fp16_path), str(out_path), quant],
        check=True,
    )
