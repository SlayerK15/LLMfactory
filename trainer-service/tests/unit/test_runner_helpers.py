"""Unit tests for the pure helpers inside training/runner.py.

The full train_and_export function is not covered here — it requires CUDA + Unsloth. The
integration test in tests/integration hits it via Modal.
"""
from __future__ import annotations

from trainer_service.training.runner import _count_tokens_approx, _format_record


def test_format_record_pretrain():
    assert _format_record({"text": "hello world"}, "pretrain") == "hello world"


def test_format_record_instruct_basic():
    out = _format_record(
        {"instruction": "Q?", "input": "", "response": "A."}, "instruct"
    )
    assert "### Instruction:\nQ?" in out
    assert "### Response:\nA." in out
    assert "### Input:" not in out


def test_format_record_instruct_with_input():
    out = _format_record(
        {"instruction": "Q?", "input": "ctx", "response": "A."}, "instruct"
    )
    assert "### Input:\nctx" in out


def test_token_count_approx_is_proportional():
    short = _count_tokens_approx(b"one two three")
    long = _count_tokens_approx(b"one two three four five six seven")
    assert long > short
