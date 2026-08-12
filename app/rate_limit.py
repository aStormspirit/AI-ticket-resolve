"""Ограничение частоты входящих запросов (spec §35).

Скользящее окно в Redis: счётчик общий для всех реплик API. При недоступности
Redis лимит не применяется — доступность сервиса важнее точности учёта, а
внешние вызовы дополнительно ограничены на уровне LLM-клиента.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, status
from redis.exceptions import RedisError

from bot.config import get_settings
from bot.storage import get_redis
from observability.metrics import rate_limited_total

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60


async def enforce_rate_limit(request: Request) -> None:
    settings = get_settings()
    limit = settings.api_requests_per_minute
    if limit <= 0:
        return

    client_host = request.client.host if request.client else "unknown"
    key = f"ratelimit:{client_host}"

    try:
        redis = get_redis()
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, WINDOW_SECONDS)
        count, _ = await pipe.execute()
    except RedisError:
        logger.warning("Redis недоступен, rate limit не применяется")
        return

    if int(count) > limit:
        rate_limited_total.labels(scope="api").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Превышен лимит {limit} запросов в минуту",
            headers={"Retry-After": str(WINDOW_SECONDS)},
        )
