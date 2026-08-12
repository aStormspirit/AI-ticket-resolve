"""Подсчёт стоимости обработки тикета.

langchain.callbacks.get_openai_callback в langchain 1.x отсутствует и всё равно
не покрывает OpenRouter, поэтому считаем по usage_metadata, которое приходит в
AIMessage, и собственной таблице цен модели.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from bot.config import get_settings
from bot.contracts import CostBreakdown, UsageRecord

# Составляющие инфраструктурных издержек (рубли/месяц), справочно для README.
INFRASTRUCTURE_COST_COMPONENTS = {
    "vector_db": 5_000,
    "observability": 3_000,
    "compute": 8_000,
    "api_gateway": 2_000,
}


def extract_usage(messages: Iterable[Any], *, step: str, model: str) -> list[UsageRecord]:
    """Собирает потребление токенов из AIMessage-ов ответа."""
    records: list[UsageRecord] = []
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            continue
        records.append(
            UsageRecord(
                step=step,
                model=model,
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            )
        )
    return records


def calculate_llm_cost_rubles(usage: Sequence[UsageRecord]) -> float:
    settings = get_settings()
    input_tokens = sum(record.input_tokens for record in usage)
    output_tokens = sum(record.output_tokens for record in usage)

    cost_usd = (
        input_tokens / 1_000_000 * settings.llm_input_price_per_1m_usd
        + output_tokens / 1_000_000 * settings.llm_output_price_per_1m_usd
    )
    return cost_usd * settings.usd_to_rub


def calculate_infrastructure_cost_per_ticket() -> float:
    settings = get_settings()
    if settings.monthly_ticket_volume <= 0:
        return 0.0
    return settings.monthly_infrastructure_cost_rubles / settings.monthly_ticket_volume


def build_cost_breakdown(usage: Sequence[UsageRecord]) -> CostBreakdown:
    llm_api = calculate_llm_cost_rubles(usage)
    infrastructure = calculate_infrastructure_cost_per_ticket()
    return CostBreakdown(
        llm_api=round(llm_api, 4),
        infrastructure=round(infrastructure, 4),
        total=round(llm_api + infrastructure, 4),
    )
