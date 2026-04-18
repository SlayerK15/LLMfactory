"""Modal app definition — runs QLoRA fine-tuning + merge + GGUF quantisation on A100.

Deploy with:  modal deploy trainer_service.training.modal_app
Invoke from:  trainer_service.adapters.compute.modal_adapter (remote call)

The heavy deps (unsloth, peft, llama-cpp) live in the Modal image only — not the local venv.
Training logic is delegated to ``trainer_service.training.runner.train_and_export`` so it can be
unit-tested independently of Modal.
"""
from __future__ import annotations

from typing import Any

import modal

APP_NAME = "collection-system-trainer"

# Image layered to cache model downloads separately from code changes
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.45,<5",
        "datasets>=3.0",
        "peft>=0.13",
        "bitsandbytes>=0.44",
        "accelerate>=1.0",
        "trl>=0.11",
        "unsloth[cu124-torch240]",
        "sentencepiece",
        "protobuf",
    )
    .pip_install("llama-cpp-python>=0.3")
    .run_commands(
        "git clone --depth=1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=OFF && cmake --build build --config Release -j",
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("trainer-cache", create_if_missing=True)


@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 2,
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
)
def train(config: dict[str, Any], train_bytes: bytes, eval_bytes: bytes) -> dict[str, Any]:
    """Entry point invoked from the local orchestrator. Delegates to runner.train_and_export."""
    from trainer_service.training.runner import train_and_export

    return train_and_export(config, train_bytes, eval_bytes, cache_dir="/cache")
