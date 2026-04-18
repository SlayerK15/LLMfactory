"""Unit tests for S3 adapter using moto to stub out AWS."""
from __future__ import annotations

from pathlib import Path

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from trainer_service.adapters.storage.s3_adapter import S3Storage
from trainer_service.core.errors import ArtifactUploadError


def test_upload_succeeds_with_moto(tmp_path: Path):
    from moto import mock_aws

    with mock_aws():
        import boto3

        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="unit-bucket")
        sink = S3Storage(bucket="unit-bucket", client=client)

        local = tmp_path / "artifact.gguf"
        local.write_bytes(b"x" * 2048)
        uri = sink.upload(local, key="trainer/run-abc/model.gguf")
        assert uri == "s3://unit-bucket/trainer/run-abc/model.gguf"

        head = client.head_object(Bucket="unit-bucket", Key="trainer/run-abc/model.gguf")
        assert head["ContentLength"] == 2048


def test_missing_bucket_raises():
    with pytest.raises(ArtifactUploadError):
        S3Storage(bucket="")
