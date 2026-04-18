"""Structured JSON logging via structlog → stdout → promtail → Loki."""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", run_id: str | None = None) -> None:
    """
    Configure structlog for JSON output.
    Call once at startup. Optionally bind run_id to all subsequent log lines.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    if run_id:
        structlog.contextvars.bind_contextvars(run_id=run_id)


def bind_run_context(run_id: str, topic: str) -> None:
    """Bind run_id and topic to all log lines for the current async context."""
    structlog.contextvars.bind_contextvars(run_id=run_id, topic=topic)


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
