"""Точка входа HTTP API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.deps import create_queue_pool
from app.routers import health, tickets
from bot.config import get_settings
from bot.storage import TicketResultStore, close_redis
from observability.logging import configure_logging, ticket_context
from observability.metrics import REGISTRY

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    app.state.result_store = TicketResultStore()
    try:
        app.state.queue = await create_queue_pool()
    except Exception:
        # Сервис поднимается и без очереди: /health/ready покажет деградацию,
        # а запросы на приём тикетов вернут 503 вместо падения контейнера.
        logger.exception("Не удалось подключиться к очереди обработки")
        app.state.queue = None

    logger.info("API запущено, модель=%s", settings.llm_model)
    try:
        yield
    finally:
        if app.state.queue is not None:
            await app.state.queue.aclose()
        await close_redis()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Support Ticket Automation",
        description="Автоматическая обработка обращений службы поддержки",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def correlation_and_timing(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        started = time.monotonic()
        with ticket_context(request_id):
            response = await call_next(request)
        elapsed = time.monotonic() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.3f}"
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    app.include_router(health.router)
    app.include_router(tickets.router)
    return app


app = create_app()
