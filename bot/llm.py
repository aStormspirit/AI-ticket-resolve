"""Клиент LLM.

Создаётся лениво: отсутствие ключа не должно ронять импорт модуля, иначе
тесты, линтеры и сборка образа требуют боевых секретов.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openrouter import ChatOpenRouter

from .config import get_settings


class LLMConfigurationError(RuntimeError):
    pass


@lru_cache
def get_rate_limiter() -> InMemoryRateLimiter:
    """Ограничивает исходящий поток запросов к LLM внутри процесса (spec §35)."""
    settings = get_settings()
    return InMemoryRateLimiter(
        requests_per_second=settings.llm_requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=settings.llm_rate_limit_burst,
    )


@lru_cache
def get_llm() -> ChatOpenRouter:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMConfigurationError(
            "Укажите OPENROUTER_API_KEY в .env (см. .env.example)"
        )

    # GPT-5 через OpenRouter: https://openrouter.ai/openai/gpt-5
    options: dict = {
        "model": settings.llm_model,
        "api_key": settings.openrouter_api_key,
        "temperature": settings.llm_temperature,
        # ChatOpenRouter ожидает timeout в миллисекундах (маппится в timeout_ms SDK).
        "timeout": int(settings.llm_request_timeout_seconds * 1000),
        # Ретраи выполняет retry_async под общим бюджетом тикета: встроенный
        # RetryConfig SDK допускает до 150 секунд на попытку и ломает SLA.
        "max_retries": 0,
        "rate_limiter": get_rate_limiter(),
    }
    # Объём внутренних размышлений — основной рычаг латентности для
    # reasoning-моделей. "none" означает не передавать параметр вовсе, что нужно
    # для моделей без поддержки reasoning.
    if settings.llm_reasoning_effort.lower() != "none":
        options["reasoning"] = {"effort": settings.llm_reasoning_effort}

    if settings.llm_base_url:
        options["base_url"] = settings.llm_base_url

    return ChatOpenRouter(**options)
