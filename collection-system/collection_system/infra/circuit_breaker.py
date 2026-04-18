"""Per-backend circuit breakers via pybreaker."""
from __future__ import annotations

import pybreaker
import structlog

from collection_system.infra.metrics import circuit_breaker_state as cb_metric

log = structlog.get_logger()


class LoggingCircuitBreakerListener(pybreaker.CircuitBreakerListener):
    def state_change(self, cb: pybreaker.CircuitBreaker, old_state: object, new_state: object) -> None:
        log.warning(
            "circuit_breaker_state_change",
            name=cb.name,
            old=str(old_state),
            new=str(new_state),
        )
        state_map = {"closed": 0, "open": 1, "half-open": 2}
        cb_metric.labels(backend=cb.name).set(state_map.get(str(new_state).lower(), -1))


def make_circuit_breaker(
    name: str,
    fail_max: int = 5,
    reset_timeout: int = 60,
) -> pybreaker.CircuitBreaker:
    return pybreaker.CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        name=name,
        listeners=[LoggingCircuitBreakerListener()],
    )


class CircuitBreakerRegistry:
    """Registry of per-backend circuit breakers."""

    def __init__(self) -> None:
        self._breakers: dict[str, pybreaker.CircuitBreaker] = {}

    def register(self, name: str, fail_max: int = 5, reset_timeout: int = 60) -> None:
        self._breakers[name] = make_circuit_breaker(name, fail_max, reset_timeout)

    def get(self, name: str) -> pybreaker.CircuitBreaker | None:
        return self._breakers.get(name)
