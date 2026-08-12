"""Конфигурация приложения через переменные окружения.

Отсутствие ключа не роняет импорт модуля: валидация происходит при создании
клиента LLM, поэтому тесты и статические проверки работают без .env.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM ---
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    llm_model: str = Field(default="openai/gpt-5", alias="LLM_MODEL")
    # Reasoning-модели тратят на внутренние размышления единицы и десятки секунд,
    # что несовместимо с бюджетом в 10 секунд на тикет. Значение по умолчанию
    # выбрано под этот бюджет; none отключает передачу параметра.
    llm_reasoning_effort: str = Field(default="minimal", alias="LLM_REASONING_EFFORT")
    # Позволяет направить трафик через внутренний шлюз или тестовый стенд.
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_request_timeout_seconds: float = Field(
        default=8.0, alias="LLM_REQUEST_TIMEOUT_SECONDS"
    )
    llm_max_retries: int = Field(default=1, alias="LLM_MAX_RETRIES")

    # --- Latency budget (spec §31) ---
    ticket_deadline_seconds: float = Field(default=10.0, alias="TICKET_DEADLINE_SECONDS")

    # --- Circuit breaker (spec §32) ---
    breaker_failure_threshold: int = Field(default=5, alias="BREAKER_FAILURE_THRESHOLD")
    breaker_recovery_seconds: float = Field(default=30.0, alias="BREAKER_RECOVERY_SECONDS")
    breaker_window_seconds: float = Field(default=60.0, alias="BREAKER_WINDOW_SECONDS")

    # --- Rate limiting (spec §35) ---
    llm_requests_per_second: float = Field(default=5.0, alias="LLM_REQUESTS_PER_SECOND")
    llm_rate_limit_burst: int = Field(default=10, alias="LLM_RATE_LIMIT_BURST")
    api_requests_per_minute: int = Field(default=120, alias="API_REQUESTS_PER_MINUTE")

    # --- Классификация ---
    confidence_threshold: float = Field(default=0.7, alias="CONFIDENCE_THRESHOLD")

    # --- Redis ---
    redis_url: RedisDsn = Field(
        default=RedisDsn("redis://localhost:6379/0"), alias="REDIS_URL"
    )
    ticket_result_ttl_seconds: int = Field(
        default=86_400, alias="TICKET_RESULT_TTL_SECONDS"
    )
    idempotency_ttl_seconds: int = Field(default=86_400, alias="IDEMPOTENCY_TTL_SECONDS")

    # --- Стоимость ---
    llm_input_price_per_1m_usd: float = Field(
        default=1.25, alias="LLM_INPUT_PRICE_PER_1M_USD"
    )
    llm_output_price_per_1m_usd: float = Field(
        default=10.0, alias="LLM_OUTPUT_PRICE_PER_1M_USD"
    )
    usd_to_rub: float = Field(default=90.0, alias="USD_TO_RUB")
    monthly_ticket_volume: int = Field(default=10_000, alias="MONTHLY_TICKET_VOLUME")
    monthly_infrastructure_cost_rubles: float = Field(
        default=18_000.0, alias="MONTHLY_INFRASTRUCTURE_COST_RUBLES"
    )

    # --- Наблюдаемость ---
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    worker_metrics_port: int = Field(default=9001, alias="WORKER_METRICS_PORT")


@lru_cache
def get_settings() -> Settings:
    return Settings()
