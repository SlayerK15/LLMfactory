"""Structlog bridge that forwards log records onto a per-request asyncio queue.

Design:
- `configure_logging()` is called once at app startup. It installs a structlog
  processor chain with `_forward_to_sse` inserted before JSONRenderer, so the
  processor sees structured fields (not the final JSON string).
- Each SSE request binds its own `asyncio.Queue` into `sse_log_queue` (a
  ContextVar). Because `asyncio.create_task` copies the parent context,
  log lines produced inside the pipeline — even from sub-tasks — are pushed
  onto the right request's queue.
"""
from __future__ import annotations

import contextvars
import logging
import sys

import structlog

sse_log_queue: contextvars.ContextVar = contextvars.ContextVar(
    "sse_log_queue", default=None
)
_bridge_installed = False


def _enqueue_line(line: str) -> None:
    queue = sse_log_queue.get()
    if queue is None:
        return
    try:
        queue.put_nowait(line)
    except Exception:
        pass


def _forward_to_sse(logger, method_name, event_dict):
    level = event_dict.get("level", method_name)
    event = event_dict.get("event", "")
    skip = {"event", "level", "timestamp", "logger"}
    fields = []
    for k, v in event_dict.items():
        if k in skip:
            continue
        fields.append(f"{k}={v}")
    line = f"[{level:<5}] {event}"
    if fields:
        line += "  " + " ".join(fields)
    _enqueue_line(line)
    return event_dict


class _Crawl4AILogHandler(logging.Handler):
    """Forward stdlib Crawl4AI logs into the active SSE queue."""

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("crawl4ai"):
            return
        level = record.levelname.lower()
        line = f"[{level:<5}] {record.name}"
        message = record.getMessage()
        if message:
            line += f"  {message}"
        _enqueue_line(line)


def configure_logging(level: str = "INFO") -> None:
    global _bridge_installed

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _forward_to_sse,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    if _bridge_installed:
        return

    handler = _Crawl4AILogHandler()
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    crawl4ai_logger = logging.getLogger("crawl4ai")
    crawl4ai_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    crawl4ai_logger.addHandler(handler)
    _bridge_installed = True
