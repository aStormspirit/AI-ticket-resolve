"""Устойчивость: бюджет времени, circuit breaker, ретраи (spec §31, §32, §36).

Бюджет считается на весь тикет, а не на отдельный вызов: только так можно
гарантировать SLA в 10 секунд при нескольких последовательных обращениях к LLM.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, TypeVar

from redis.exceptions import RedisError

from observability.metrics import circuit_breaker_state, circuit_breaker_trips_total

from .config import get_settings
from .storage import get_redis

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DeadlineExceeded(Exception):
    """Бюджет времени на обработку тикета исчерпан."""


class StepTimeout(TimeoutError):
    """Отдельный вызов зависимости не уложился в собственный таймаут.

    В отличие от DeadlineExceeded это признак проблемы на стороне зависимости,
    поэтому такой отказ учитывается circuit breaker-ом.
    """


class CircuitBreakerOpen(Exception):
    """Внешняя зависимость помечена недоступной, вызов не производится."""

    def __init__(self, target: str) -> None:
        super().__init__(f"Circuit breaker '{target}' разомкнут")
        self.target = target


class CircuitState(str, Enum):
    CLOSED = "closed"
    HALF_OPEN = "half_open"
    OPEN = "open"


_STATE_METRIC_VALUE = {
    CircuitState.CLOSED: 0.0,
    CircuitState.HALF_OPEN: 1.0,
    CircuitState.OPEN: 2.0,
}


@dataclass
class Deadline:
    """Общий бюджет времени на обработку одного тикета."""

    budget_seconds: float
    started_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_settings(cls) -> "Deadline":
        return cls(budget_seconds=get_settings().ticket_deadline_seconds)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining(self) -> float:
        return self.budget_seconds - self.elapsed

    @property
    def expired(self) -> bool:
        return self.remaining <= 0

    def check(self) -> None:
        if self.expired:
            raise DeadlineExceeded(
                f"Бюджет {self.budget_seconds:.1f}s исчерпан "
                f"(прошло {self.elapsed:.1f}s)"
            )

    async def run(
        self,
        awaitable: Awaitable[T],
        *,
        min_seconds: float = 0.5,
        max_seconds: float | None = None,
    ) -> T:
        """Выполняет операцию в пределах остатка бюджета.

        max_seconds ограничивает один шаг, чтобы он не израсходовал весь бюджет
        тикета: таймаут самого HTTP-клиента срабатывает не во всех сценариях
        (например, при «молчащем» соединении), поэтому ограничение ставится здесь.
        """
        self.check()
        remaining = self.remaining
        if remaining < min_seconds:
            raise DeadlineExceeded(
                f"Осталось {remaining:.2f}s — недостаточно для следующего шага"
            )

        limit = remaining if max_seconds is None else min(remaining, max_seconds)
        try:
            async with asyncio.timeout(limit):
                return await awaitable
        except TimeoutError as exc:
            if self.expired:
                raise DeadlineExceeded(
                    f"Шаг не уложился в оставшиеся {remaining:.2f}s"
                ) from exc
            raise StepTimeout(f"Вызов не уложился в {limit:.2f}s") from exc


class _LocalBreakerState:
    """Фолбэк на случай недоступности Redis: состояние в пределах процесса."""

    def __init__(self) -> None:
        self.failures = 0
        self.window_started_at = 0.0
        self.opened_at: float | None = None


class CircuitBreaker:
    """Circuit breaker с состоянием в Redis, общим для всех реплик.

    При недоступности Redis деградирует до состояния в памяти процесса, а не
    отказывает: иначе сбой вспомогательного хранилища блокировал бы обработку.
    """

    def __init__(
        self,
        target: str,
        *,
        failure_threshold: int | None = None,
        recovery_seconds: float | None = None,
        window_seconds: float | None = None,
    ) -> None:
        settings = get_settings()
        self.target = target
        self.failure_threshold = failure_threshold or settings.breaker_failure_threshold
        self.recovery_seconds = recovery_seconds or settings.breaker_recovery_seconds
        self.window_seconds = window_seconds or settings.breaker_window_seconds
        self._local = _LocalBreakerState()
        self._lock = asyncio.Lock()

    @property
    def _failures_key(self) -> str:
        return f"cb:{self.target}:failures"

    @property
    def _opened_key(self) -> str:
        return f"cb:{self.target}:opened_at"

    @property
    def _probe_key(self) -> str:
        return f"cb:{self.target}:probe"

    async def state(self) -> CircuitState:
        opened_at = await self._get_opened_at()
        if opened_at is None:
            return CircuitState.CLOSED
        if time.time() - opened_at >= self.recovery_seconds:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    async def acquire(self) -> None:
        """Бросает CircuitBreakerOpen, если вызов сейчас запрещён."""
        state = await self.state()
        self._publish_state(state)

        if state is CircuitState.CLOSED:
            return

        if state is CircuitState.OPEN:
            raise CircuitBreakerOpen(self.target)

        # HALF_OPEN: пропускаем один пробный вызов.
        if not await self._claim_probe():
            raise CircuitBreakerOpen(self.target)

    async def record_success(self) -> None:
        try:
            redis = get_redis()
            await redis.delete(self._failures_key, self._opened_key, self._probe_key)
        except RedisError:
            async with self._lock:
                self._local.failures = 0
                self._local.opened_at = None
        self._publish_state(CircuitState.CLOSED)

    async def record_failure(self) -> None:
        if await self.state() is CircuitState.HALF_OPEN:
            # Пробный вызов не удался — отсчёт восстановления начинается заново.
            await self._open(refresh=True)
            return

        failures = await self._increment_failures()
        if failures >= self.failure_threshold:
            await self._open()

    async def _open(self, *, refresh: bool = False) -> None:
        """Размыкает breaker. Учитывается только переход, а не каждая ошибка."""
        now = time.time()
        ttl = int(self.recovery_seconds * 2) + 1

        try:
            redis = get_redis()
            transitioned = await redis.set(
                self._opened_key, now, ex=ttl, nx=not refresh
            )
            if refresh or transitioned:
                await redis.delete(self._probe_key)
        except RedisError:
            async with self._lock:
                transitioned = refresh or self._local.opened_at is None
                if transitioned:
                    self._local.opened_at = now

        if not (refresh or transitioned):
            return

        circuit_breaker_trips_total.labels(target=self.target).inc()
        self._publish_state(CircuitState.OPEN)
        logger.error(
            "Circuit breaker '%s' разомкнут на %.0fs после %d ошибок",
            self.target,
            self.recovery_seconds,
            self.failure_threshold,
        )

    async def _increment_failures(self) -> int:
        try:
            redis = get_redis()
            pipe = redis.pipeline()
            pipe.incr(self._failures_key)
            pipe.expire(self._failures_key, int(self.window_seconds) + 1)
            failures, _ = await pipe.execute()
            return int(failures)
        except RedisError:
            async with self._lock:
                now = time.monotonic()
                if now - self._local.window_started_at > self.window_seconds:
                    self._local.window_started_at = now
                    self._local.failures = 0
                self._local.failures += 1
                return self._local.failures

    async def _get_opened_at(self) -> float | None:
        try:
            raw = await get_redis().get(self._opened_key)
        except RedisError:
            return self._local.opened_at
        return float(raw) if raw is not None else None

    async def _claim_probe(self) -> bool:
        """Только один вызов проходит в half-open состоянии."""
        try:
            acquired = await get_redis().set(
                self._probe_key, "1", nx=True, ex=int(self.recovery_seconds) + 1
            )
            return bool(acquired)
        except RedisError:
            return True

    def _publish_state(self, state: CircuitState) -> None:
        circuit_breaker_state.labels(target=self.target).set(_STATE_METRIC_VALUE[state])

    async def call(self, factory: Callable[[], Awaitable[T]]) -> T:
        """Выполняет операцию под защитой breaker-а."""
        await self.acquire()
        try:
            result = await factory()
        except (DeadlineExceeded, asyncio.CancelledError):
            # Исчерпание бюджета — свойство тикета, а не признак недоступности
            # зависимости, поэтому breaker не размыкаем.
            raise
        except Exception:
            await self.record_failure()
            raise
        await self.record_success()
        return result


async def retry_async(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 2,
    base_delay: float = 0.2,
    max_delay: float = 2.0,
    deadline: Deadline | None = None,
    min_attempt_seconds: float = 0.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Ретрай с экспоненциальной задержкой и джиттером, ограниченный дедлайном.

    min_attempt_seconds — минимум времени, который должен остаться на новую
    попытку. Без этого условия повтор запускается на остатках бюджета, заведомо
    не успевает и лишь задерживает эскалацию.
    """
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except (DeadlineExceeded, CircuitBreakerOpen):
            raise
        except retry_on as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            delay += random.uniform(0, delay / 2)
            if deadline is not None and deadline.remaining <= delay + min_attempt_seconds:
                logger.warning(
                    "Повтор отменён: осталось %.2fs, попытке нужно не менее %.2fs",
                    deadline.remaining,
                    delay + min_attempt_seconds,
                )
                break
            logger.warning(
                "Попытка %d/%d не удалась (%s), повтор через %.2fs",
                attempt,
                attempts,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


llm_breaker = CircuitBreaker("llm")
knowledge_base_breaker = CircuitBreaker("knowledge_base")
orders_api_breaker = CircuitBreaker("orders_api")
