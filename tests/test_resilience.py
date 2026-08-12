from __future__ import annotations

import asyncio

import pytest

from bot.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
    Deadline,
    DeadlineExceeded,
    StepTimeout,
    retry_async,
)

pytestmark = pytest.mark.asyncio


async def test_deadline_reports_remaining_budget() -> None:
    deadline = Deadline(budget_seconds=5)

    assert deadline.expired is False
    assert 4.5 < deadline.remaining <= 5


async def test_deadline_raises_when_budget_spent() -> None:
    deadline = Deadline(budget_seconds=0.05)
    await asyncio.sleep(0.06)

    with pytest.raises(DeadlineExceeded):
        deadline.check()


async def test_deadline_interrupts_slow_operation() -> None:
    deadline = Deadline(budget_seconds=0.2)

    with pytest.raises(DeadlineExceeded):
        await deadline.run(asyncio.sleep(1), min_seconds=0.01)


async def test_step_cap_does_not_consume_whole_budget() -> None:
    """Один шаг не должен съедать весь бюджет тикета."""
    deadline = Deadline(budget_seconds=5)

    with pytest.raises(StepTimeout):
        await deadline.run(asyncio.sleep(2), max_seconds=0.1)

    assert deadline.remaining > 4


async def test_hanging_dependency_opens_breaker() -> None:
    """Зависший вызов — отказ зависимости, а не свойство тикета."""
    breaker = CircuitBreaker("test-hang", failure_threshold=2, recovery_seconds=60)

    async def hanging() -> None:
        deadline = Deadline(budget_seconds=5)
        await deadline.run(asyncio.sleep(1), max_seconds=0.05)

    for _ in range(2):
        with pytest.raises(StepTimeout):
            await breaker.call(hanging)

    assert await breaker.state() is CircuitState.OPEN


async def test_deadline_refuses_step_without_time_left() -> None:
    deadline = Deadline(budget_seconds=0.3)
    operation = asyncio.sleep(0)

    with pytest.raises(DeadlineExceeded):
        await deadline.run(operation, min_seconds=1.0)

    operation.close()


async def test_breaker_opens_after_threshold() -> None:
    breaker = CircuitBreaker("test-open", failure_threshold=3, recovery_seconds=60)

    async def failing() -> None:
        raise RuntimeError("boom")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(failing)

    assert await breaker.state() is CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        await breaker.call(failing)


async def test_breaker_allows_single_probe_after_recovery() -> None:
    breaker = CircuitBreaker("test-probe", failure_threshold=1, recovery_seconds=0.1)

    async def failing() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await breaker.call(failing)
    assert await breaker.state() is CircuitState.OPEN

    await asyncio.sleep(0.15)
    assert await breaker.state() is CircuitState.HALF_OPEN

    await breaker.call(_succeed)
    assert await breaker.state() is CircuitState.CLOSED


async def test_breaker_counts_only_the_transition() -> None:
    """Повторные отказы при уже разомкнутом breaker не считаются новыми срабатываниями."""
    from observability.metrics import REGISTRY

    breaker = CircuitBreaker("test-trips", failure_threshold=1, recovery_seconds=60)
    before = (
        REGISTRY.get_sample_value(
            "support_circuit_breaker_trips_total", {"target": "test-trips"}
        )
        or 0.0
    )

    for _ in range(3):
        await breaker.record_failure()

    after = REGISTRY.get_sample_value(
        "support_circuit_breaker_trips_total", {"target": "test-trips"}
    )
    assert after == before + 1


async def test_failed_probe_restarts_recovery_window() -> None:
    breaker = CircuitBreaker("test-reopen", failure_threshold=1, recovery_seconds=0.2)

    await breaker.record_failure()
    await asyncio.sleep(0.25)
    assert await breaker.state() is CircuitState.HALF_OPEN

    await breaker.record_failure()

    assert await breaker.state() is CircuitState.OPEN


async def test_breaker_ignores_deadline_errors() -> None:
    """Исчерпание бюджета — свойство тикета, а не отказ зависимости."""
    breaker = CircuitBreaker("test-deadline", failure_threshold=1, recovery_seconds=60)

    async def timing_out() -> None:
        raise DeadlineExceeded("no time left")

    with pytest.raises(DeadlineExceeded):
        await breaker.call(timing_out)

    assert await breaker.state() is CircuitState.CLOSED


async def test_breaker_falls_back_to_local_state_without_redis(fake_redis) -> None:
    fake_redis.fail = True
    breaker = CircuitBreaker("test-local", failure_threshold=2, recovery_seconds=60)

    async def failing() -> None:
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing)

    assert await breaker.state() is CircuitState.OPEN


async def test_retry_succeeds_on_second_attempt() -> None:
    attempts = {"count": 0}

    async def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert await retry_async(flaky, attempts=3, base_delay=0.01) == "ok"
    assert attempts["count"] == 2


async def test_retry_skipped_when_budget_cannot_fit_attempt() -> None:
    """Повтор на остатках бюджета только задерживает эскалацию."""
    attempts = {"count": 0}
    deadline = Deadline(budget_seconds=1.0)

    async def failing() -> None:
        attempts["count"] += 1
        raise RuntimeError("slow dependency")

    with pytest.raises(RuntimeError):
        await retry_async(
            failing,
            attempts=3,
            base_delay=0.01,
            deadline=deadline,
            min_attempt_seconds=8.0,
        )

    assert attempts["count"] == 1
    assert deadline.remaining > 0.9


async def test_retry_does_not_retry_open_breaker() -> None:
    attempts = {"count": 0}

    async def blocked() -> None:
        attempts["count"] += 1
        raise CircuitBreakerOpen("llm")

    with pytest.raises(CircuitBreakerOpen):
        await retry_async(blocked, attempts=3, base_delay=0.01)

    assert attempts["count"] == 1


async def _succeed() -> str:
    return "ok"
