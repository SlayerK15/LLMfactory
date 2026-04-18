"""S3 artifact uploader. Thin wrapper over boto3 with retries."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from trainer_service.core.errors import ArtifactUploadError

log = structlog.get_logger()


class S3Storage:
    def __init__(self, bucket: str, region: str | None = None, client: Any | None = None) -> None:
        if not bucket:
            raise ArtifactUploadError("s3 bucket is required")
        self._bucket = bucket
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ArtifactUploadError("boto3 not installed") from exc
            self._client = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=2.0, max=20.0), reraise=True)
    def upload(self, local_path: Path, key: str, content_type: str = "application/octet-stream") -> str:
        try:
            self._client.upload_file(
                str(local_path),
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        except Exception as exc:
            raise ArtifactUploadError(f"S3 upload failed: {exc}") from exc
        uri = f"s3://{self._bucket}/{key}"
        log.info("s3.upload_done", uri=uri, bytes=local_path.stat().st_size)
        return uri
