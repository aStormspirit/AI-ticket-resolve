"""Фоновый воркер обработки тикетов на arq.

Обработка вынесена из HTTP-цикла: приём обращения не зависит от латентности
LLM, а количество параллельных обработок регулируется настройками воркера.
"""

from __future__ import annotations

import logging
from typing import Any

from arq.connections import RedisSettings
from prometheus_client import start_http_server

from bot.config import get_settings
from bot.contracts import TicketResult
from bot.pipeline import process_ticket
from bot.storage import TicketResultStore, close_redis
from observability.logging import configure_logging
from observability.metrics import REGISTRY, queue_depth

logger = logging.getLogger(__name__)

QUEUE_NAME = "arq:queue"


async def process_ticket_job(ctx: dict[str, Any], text: str, ticket_id: str) -> dict:
    """Задача очереди: полный цикл обработки одного обращения."""
    store: TicketResultStore = ctx["result_store"]

    existing = await store.get(ticket_id)
    if existing is not None:
        logger.info("Тикет %s уже обработан, повторная обработка пропущена", ticket_id)
        return existing.model_dump(mode="json")

    result: TicketResult = await process_ticket(text, ticket_id)
    await store.save(result)
    await _publish_queue_depth(ctx)
    return result.model_dump(mode="json")


async def _publish_queue_depth(ctx: dict[str, Any]) -> None:
    try:
        depth = await ctx["redis"].zcard(QUEUE_NAME)
        queue_depth.labels(queue=QUEUE_NAME).set(depth)
    except Exception:
        logger.debug("Не удалось получить глубину очереди", exc_info=True)


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    ctx["result_store"] = TicketResultStore()

    # Отдельный порт метрик: воркер и API живут в разных процессах и
    # скрапятся Prometheus независимо.
    start_http_server(settings.worker_metrics_port, registry=REGISTRY)
    logger.info(
        "Воркер запущен, метрики на :%d, модель=%s",
        settings.worker_metrics_port,
        settings.llm_model,
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    await close_redis()
    logger.info("Воркер остановлен")


class WorkerSettings:
    functions = [process_ticket_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(str(get_settings().redis_url))
    max_jobs = 10
    job_timeout = 60
    keep_result = 3600
    # Повторы задачи безопасны: действия инструментов защищены ключами
    # идемпотентности (spec §34).
    max_tries = 2
