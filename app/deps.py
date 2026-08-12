"""Зависимости FastAPI: пул очереди и хранилище результатов."""

from __future__ import annotations

from typing import Annotated

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from fastapi import Depends, HTTPException, Request, status

from bot.config import get_settings
from bot.storage import TicketResultStore

QUEUE_NAME = "arq:queue"


async def create_queue_pool() -> ArqRedis:
    settings = get_settings()
    return await create_pool(RedisSettings.from_dsn(str(settings.redis_url)))


def get_queue(request: Request) -> ArqRedis:
    pool: ArqRedis | None = getattr(request.app.state, "queue", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Очередь обработки недоступна",
        )
    return pool


def get_result_store(request: Request) -> TicketResultStore:
    return request.app.state.result_store


QueueDep = Annotated[ArqRedis, Depends(get_queue)]
ResultStoreDep = Annotated[TicketResultStore, Depends(get_result_store)]
