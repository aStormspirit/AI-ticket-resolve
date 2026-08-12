"""Доступ к Redis и хранилище результатов тикетов.

Redis — единственное внешнее состояние сервиса: результаты обработки, ключи
идемпотентности, состояние circuit breaker и очередь задач arq.
"""

from __future__ import annotations

import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import get_settings
from .contracts import TicketResult

logger = logging.getLogger(__name__)

_client: Redis | None = None

RESULT_KEY_PREFIX = "ticket:result:"


def get_redis() -> Redis:
    """Возвращает разделяемый пул соединений."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = Redis.from_url(
            str(settings.redis_url),
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def ping_redis() -> bool:
    try:
        return bool(await get_redis().ping())
    except RedisError:
        return False


class TicketResultStore:
    """Результаты обработки с TTL: воркер пишет, API читает."""

    def __init__(self, redis: Redis | None = None, ttl_seconds: int | None = None) -> None:
        self._redis = redis or get_redis()
        self._ttl = ttl_seconds or get_settings().ticket_result_ttl_seconds

    @staticmethod
    def _key(ticket_id: str) -> str:
        return f"{RESULT_KEY_PREFIX}{ticket_id}"

    async def save(self, result: TicketResult) -> None:
        try:
            await self._redis.set(
                self._key(result.ticket_id),
                result.model_dump_json(),
                ex=self._ttl,
            )
        except RedisError:
            # Потеря результата не должна ронять уже выполненную обработку.
            logger.exception("Не удалось сохранить результат тикета в Redis")

    async def get(self, ticket_id: str) -> TicketResult | None:
        try:
            raw = await self._redis.get(self._key(ticket_id))
        except RedisError:
            logger.exception("Не удалось прочитать результат тикета из Redis")
            return None
        if raw is None:
            return None
        return TicketResult.model_validate_json(raw)

    async def exists(self, ticket_id: str) -> bool:
        try:
            return bool(await self._redis.exists(self._key(ticket_id)))
        except RedisError:
            return False
