"""Идемпотентность вызовов инструментов (spec §34).

Повторный вызов при ретрае не должен приводить к повторному действию: клиент не
получит два письма о сбросе пароля. Результат первого успешного вызова
сохраняется в Redis и возвращается при последующих обращениях с тем же ключом.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextvars import ContextVar
from typing import Any, Awaitable, Callable

from redis.exceptions import RedisError

from .config import get_settings
from .storage import get_redis

logger = logging.getLogger(__name__)

KEY_PREFIX = "idem:"

# Инструменты не получают ticket_id в аргументах, поэтому берут его из контекста
# исполнения — так ключ идемпотентности остаётся привязанным к обращению.
_scope: ContextVar[str] = ContextVar("idempotency_scope", default="global")

# Фолбэк при недоступности Redis: гарантия в пределах процесса.
_local_cache: dict[str, dict[str, Any]] = {}


def set_scope(scope: str) -> object:
    return _scope.set(scope)


def reset_scope(token: object) -> None:
    _scope.reset(token)  # type: ignore[arg-type]


def current_scope() -> str:
    return _scope.get()


def build_key(tool: str, **arguments: Any) -> str:
    """Ключ = область (тикет) + имя инструмента + нормализованные аргументы."""
    payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{KEY_PREFIX}{current_scope()}:{tool}:{digest}"


async def run_once(
    tool: str,
    arguments: dict[str, Any],
    action: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    """Выполняет действие не более одного раза на комбинацию (тикет, инструмент, аргументы).

    В результат повтора добавляется флаг idempotent_replay, чтобы наблюдаемость
    отличала фактическое действие от воспроизведённого ответа.
    """
    key = build_key(tool, **arguments)

    cached = await _get(key)
    if cached is not None:
        logger.info("Повторный вызов %s подавлен ключом идемпотентности", tool)
        return {**cached, "idempotent_replay": True}

    result = await action()

    # Кэшируем только успешные действия: неуспех должен допускать повтор.
    if result.get("success"):
        await _put(key, result)

    return result


async def _get(key: str) -> dict[str, Any] | None:
    try:
        raw = await get_redis().get(key)
    except RedisError:
        logger.warning("Redis недоступен, идемпотентность работает в пределах процесса")
        return _local_cache.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def _put(key: str, value: dict[str, Any]) -> None:
    ttl = get_settings().idempotency_ttl_seconds
    try:
        await get_redis().set(
            key, json.dumps(value, ensure_ascii=False, default=str), ex=ttl
        )
    except RedisError:
        _local_cache[key] = value
