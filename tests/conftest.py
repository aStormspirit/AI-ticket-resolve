"""Общие фикстуры: подмена Redis и изоляция настроек."""

from __future__ import annotations

import time
from typing import Any

import pytest

import bot.storage
from bot.config import Settings, get_settings


class FakeRedis:
    """Минимальная in-memory замена Redis для тестов."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.expiry: dict[str, float] = {}
        self.fail = False

    # --- служебное ---

    def _check(self) -> None:
        if self.fail:
            from redis.exceptions import ConnectionError as RedisConnectionError

            raise RedisConnectionError("fake redis is down")

    def _expired(self, key: str) -> bool:
        deadline = self.expiry.get(key)
        if deadline is not None and deadline <= time.time():
            self.data.pop(key, None)
            self.expiry.pop(key, None)
            return True
        return False

    # --- команды ---

    async def ping(self) -> bool:
        self._check()
        return True

    async def get(self, key: str) -> Any:
        self._check()
        if self._expired(key):
            return None
        return self.data.get(key)

    async def set(
        self, key: str, value: Any, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        self._check()
        self._expired(key)
        if nx and key in self.data:
            return None
        self.data[key] = str(value)
        if ex is not None:
            self.expiry[key] = time.time() + ex
        return True

    async def delete(self, *keys: str) -> int:
        self._check()
        removed = 0
        for key in keys:
            removed += 1 if self.data.pop(key, None) is not None else 0
            self.expiry.pop(key, None)
        return removed

    async def exists(self, key: str) -> int:
        self._check()
        if self._expired(key):
            return 0
        return 1 if key in self.data else 0

    async def incr(self, key: str) -> int:
        self._check()
        self._expired(key)
        value = int(self.data.get(key, 0)) + 1
        self.data[key] = str(value)
        return value

    async def expire(self, key: str, ttl: int) -> bool:
        self._check()
        if key not in self.data:
            return False
        self.expiry[key] = time.time() + ttl
        return True

    async def zcard(self, key: str) -> int:
        self._check()
        return 0

    async def aclose(self) -> None:
        return None

    def pipeline(self) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    """Команды копятся синхронно и выполняются в execute(), как в redis-py."""

    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._commands: list[tuple[str, tuple, dict]] = []

    def incr(self, key: str) -> "FakePipeline":
        self._commands.append(("incr", (key,), {}))
        return self

    def expire(self, key: str, ttl: int) -> "FakePipeline":
        self._commands.append(("expire", (key, ttl), {}))
        return self

    def get(self, key: str) -> "FakePipeline":
        self._commands.append(("get", (key,), {}))
        return self

    async def execute(self) -> list[Any]:
        self._redis._check()
        results = []
        for name, args, kwargs in self._commands:
            results.append(await getattr(self._redis, name)(*args, **kwargs))
        self._commands.clear()
        return results


TEST_ENV = {
    "OPENROUTER_API_KEY": "test-key",
    "LLM_MODEL": "openai/gpt-5",
    "CONFIDENCE_THRESHOLD": "0.7",
    "TICKET_DEADLINE_SECONDS": "10",
    "BREAKER_FAILURE_THRESHOLD": "3",
    "BREAKER_RECOVERY_SECONDS": "1",
    "BREAKER_WINDOW_SECONDS": "60",
    "API_REQUESTS_PER_MINUTE": "1000",
    "MONTHLY_TICKET_VOLUME": "10000",
    "MONTHLY_INFRASTRUCTURE_COST_RUBLES": "18000",
    "LLM_INPUT_PRICE_PER_1M_USD": "1.25",
    "LLM_OUTPUT_PRICE_PER_1M_USD": "10",
    "USD_TO_RUB": "90",
    "REDIS_URL": "redis://localhost:6379/15",
}


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Детерминированные настройки: переменные окружения перекрывают .env."""
    for key, value in TEST_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    """Все модули берут клиента через bot.storage._client, поэтому подменяем его."""
    redis = FakeRedis()
    monkeypatch.setattr(bot.storage, "_client", redis, raising=False)
    return redis


@pytest.fixture(autouse=True)
def reset_metric_state() -> None:
    """Сбрасывает кэши, зависящие от настроек, между тестами."""
    from bot import knowledge_base, llm, ticket_resolver

    llm.get_llm.cache_clear()
    llm.get_rate_limiter.cache_clear()
    knowledge_base.get_knowledge_base.cache_clear()
    ticket_resolver._get_agent.cache_clear()
    yield
