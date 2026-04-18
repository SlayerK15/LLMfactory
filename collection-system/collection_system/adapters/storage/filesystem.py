"""Filesystem adapter — raw document content storage."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog

from collection_system.core.errors import StorageError
from collection_system.core.models import RunManifest, ScrapedDoc

log = structlog.get_logger()


class FilesystemAdapter:
    """
    Stores raw scraped documents on the local filesystem.
    Structure: data/runs/{run_id}/docs/{doc_id}.md + .meta.json
    All writes go through asyncio.to_thread so the event loop is never blocked.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def run_dir(self, run_id: str) -> Path:
        return self.data_dir / "runs" / run_id

    def docs_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "docs"

    async def write_doc(self, doc: ScrapedDoc) -> Path:
        """Write markdown + metadata. Returns path to the .md file."""
        self.ensure_run_dirs(doc.run_id)
        md_path = self.docs_dir(doc.run_id) / f"{doc.id}.md"
        meta_path = self.docs_dir(doc.run_id) / f"{doc.id}.meta.json"

        meta = {
            "id": doc.id,
            "run_id": doc.run_id,
            "url_id": doc.url_id,
            "url": doc.url,
            "title": doc.title,
            "content_hash": doc.content_hash,
            "token_count": doc.token_count,
            "extraction_confidence": doc.extraction_confidence,
            "scraped_at": doc.scraped_at.isoformat(),
            "scrape_duration_ms": doc.scrape_duration_ms,
        }

        def _write() -> None:
            md_path.write_text(doc.markdown, encoding="utf-8")
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise StorageError(f"write_doc failed for {doc.id}") from exc

        log.debug("fs.write_doc", run_id=doc.run_id, doc_id=doc.id, bytes=len(doc.markdown))
        return md_path

    async def write_manifest(self, manifest: RunManifest) -> None:
        self.ensure_run_dirs(manifest.run_id)
        path = self.run_dir(manifest.run_id) / "manifest.json"
        payload = manifest.model_dump(mode="json")

        try:
            await asyncio.to_thread(
                path.write_text, json.dumps(payload, indent=2), "utf-8"
            )
        except OSError as exc:
            raise StorageError(f"write_manifest failed for {manifest.run_id}") from exc

        log.info("fs.write_manifest", run_id=manifest.run_id, path=str(path))

    async def write_metrics(self, run_id: str, metrics: dict) -> None:
        self.ensure_run_dirs(run_id)
        path = self.run_dir(run_id) / "metrics.json"
        try:
            await asyncio.to_thread(
                path.write_text, json.dumps(metrics, indent=2), "utf-8"
            )
        except OSError as exc:
            raise StorageError(f"write_metrics failed for {run_id}") from exc

    async def read_doc(self, run_id: str, doc_id: str) -> ScrapedDoc:
        md_path = self.docs_dir(run_id) / f"{doc_id}.md"
        meta_path = self.docs_dir(run_id) / f"{doc_id}.meta.json"

        if not md_path.exists() or not meta_path.exists():
            raise StorageError(f"Missing doc files for {doc_id} in run {run_id}")

        def _read() -> tuple[str, dict]:
            md = md_path.read_text(encoding="utf-8")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return md, meta

        try:
            markdown, meta = await asyncio.to_thread(_read)
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError(f"read_doc failed for {doc_id}") from exc

        from datetime import datetime

        return ScrapedDoc(
            id=meta["id"],
            run_id=meta["run_id"],
            url_id=meta["url_id"],
            url=meta["url"],
            title=meta.get("title"),
            markdown=markdown,
            content_hash=meta["content_hash"],
            token_count=meta.get("token_count", 0),
            extraction_confidence=meta.get("extraction_confidence", 1.0),
            scraped_at=datetime.fromisoformat(meta["scraped_at"]),
            scrape_duration_ms=meta.get("scrape_duration_ms", 0),
            path=md_path,
        )

    def ensure_run_dirs(self, run_id: str) -> None:
        self.docs_dir(run_id).mkdir(parents=True, exist_ok=True)
        (self.run_dir(run_id) / "checkpoints").mkdir(exist_ok=True)
