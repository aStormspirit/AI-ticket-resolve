"""Классификация намерения обращения."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from observability.cost import extract_usage
from observability.metrics import llm_errors_total, llm_request_duration

from .config import get_settings
from .contracts import INTENT_DESCRIPTIONS, Intent, UsageRecord
from .llm import get_llm
from .resilience import Deadline, llm_breaker, retry_async

STEP = "classification"


class IntentClassification(BaseModel):
    intent: Intent = Field(description="Категория намерения пользователя")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Уверенность в классификации от 0 до 1"
    )
    reasoning: str = Field(description="Объяснение почему выбрана эта категория")


@dataclass
class ClassificationResult:
    classification: IntentClassification
    usage: list[UsageRecord] = field(default_factory=list)


SYSTEM_PROMPT = """Ты классификатор намерений для службы поддержки.

Доступные категории:
{intents}

Проанализируй запрос пользователя и определи его намерение.
Поле confidence — честная оценка твоей уверенности от 0 до 1. Не занижай и не
завышай её и не подменяй категорию на general_inquiry при низкой уверенности:
порог автоматической обработки применяется отдельно на стороне системы.
Категорию general_inquiry выбирай только когда запрос действительно не подходит
ни под одну из остальных категорий."""


@lru_cache
def _get_chain() -> Runnable:
    # include_raw=True сохраняет исходный AIMessage, из которого берётся
    # usage_metadata для подсчёта стоимости.
    structured_llm = get_llm().with_structured_output(
        IntentClassification, include_raw=True
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", "{ticket_text}")]
    )
    return prompt | structured_llm


def _format_intents() -> str:
    return "\n".join(
        f"- {intent.value}: {description}"
        for intent, description in INTENT_DESCRIPTIONS.items()
    )


async def classify_intent(
    ticket_text: str, deadline: Deadline | None = None
) -> ClassificationResult:
    """Определяет намерение пользователя.

    Бросает исключение при недоступности модели — вызывающая сторона обязана
    перевести тикет в очередь операторов (spec §32).
    """
    settings = get_settings()
    deadline = deadline or Deadline.from_settings()
    chain = _get_chain()
    payload = {"ticket_text": ticket_text, "intents": _format_intents()}

    async def _invoke() -> dict:
        started = time.monotonic()
        try:
            result = await deadline.run(
                chain.ainvoke(payload),
                max_seconds=settings.llm_request_timeout_seconds,
            )
        except Exception as exc:
            llm_errors_total.labels(step=STEP, error_type=type(exc).__name__).inc()
            raise
        finally:
            llm_request_duration.labels(step=STEP).observe(time.monotonic() - started)
        return result

    raw_result = await llm_breaker.call(
        lambda: retry_async(
            _invoke,
            attempts=settings.llm_max_retries + 1,
            deadline=deadline,
            min_attempt_seconds=settings.llm_request_timeout_seconds,
        )
    )

    parsing_error = raw_result.get("parsing_error")
    parsed = raw_result.get("parsed")
    if parsing_error is not None or parsed is None:
        raise ValueError(f"Модель вернула невалидный ответ классификации: {parsing_error}")

    raw_message = raw_result.get("raw")
    usage = extract_usage(
        [raw_message] if raw_message is not None else [],
        step=STEP,
        model=settings.llm_model,
    )
    return ClassificationResult(classification=parsed, usage=usage)
