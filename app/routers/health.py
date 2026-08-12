"""Проверки жизнеспособности и готовности."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.schemas import HealthResponse
from bot.resilience import llm_breaker
from bot.storage import ping_redis

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse, summary="Процесс жив")
async def live() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse, summary="Готовность принимать нагрузку")
async def ready(response: Response) -> HealthResponse:
    redis_ok = await ping_redis()
    breaker_state = await llm_breaker.state()

    # Разомкнутый breaker не делает сервис неготовым: тикеты продолжают
    # приниматься и эскалируются операторам (spec §32).
    if not redis_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        redis=redis_ok,
        llm_breaker=breaker_state.value,
    )
