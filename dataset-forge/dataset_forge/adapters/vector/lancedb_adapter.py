"""LanceDB sink for chunk embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

from dataset_forge.core.errors import EmbedError

log = structlog.get_logger()


class LanceDBSink:
    def __init__(self, db_path: Path, table_name: str) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise EmbedError("lancedb not installed") from exc
        db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(db_path))
        self._table_name = table_name

    def upsert(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        if self._table_name in self._db.table_names():
            table = self._db.open_table(self._table_name)
            table.add(records)
        else:
            self._db.create_table(self._table_name, records)
        log.info("lancedb.upsert", table=self._table_name, n=len(records))
        return len(records)
