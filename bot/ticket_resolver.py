"""Автоматическое разрешение тикета агентом с инструментами."""

from __future__ import annotations

import logging
import re
import time
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, EmailStr, field_validator

from observability.cost import extract_usage
from observability.metrics import llm_errors_total, llm_request_duration

from .config import get_settings
from .contracts import (
    ADVISORY_INTENTS,
    RESOLVING_TOOL_BY_INTENT,
    Intent,
    ResolutionAttempt,
    ToolCallRecord,
    UsageRecord,
)
from .idempotency import run_once
from .integrations import APIError, OrderNotFoundError, fetch_order_status, send_reset_link_via_api
from .knowledge_base import search as search_kb
from .llm import get_llm
from .resilience import Deadline, llm_breaker, orders_api_breaker, retry_async

logger = logging.getLogger(__name__)

STEP = "resolution"
KNOWLEDGE_BASE_TOOL = "search_knowledge_base"

# Инструменты пишут структурный след сюда: проверять факт сбоя разбором текста
# ToolMessage ненадёжно — формат сериализации не является контрактом.
_tool_records: ContextVar[list[ToolCallRecord]] = ContextVar("tool_records")


def _record(record: ToolCallRecord) -> None:
    try:
        _tool_records.get().append(record)
    except LookupError:
        logger.debug("Вызов инструмента %s вне контекста обработки", record.tool)


class PasswordResetInput(BaseModel):
    """Схема валидации для сброса пароля."""

    email: EmailStr


class OrderStatusInput(BaseModel):
    """Схема валидации для проверки заказа."""

    order_id: str

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, value: str) -> str:
        if not re.match(r"^ORD-\d{6}$", value):
            raise ValueError("Order ID must be in format ORD-XXXXXX")
        return value


class KnowledgeBaseInput(BaseModel):
    """Схема валидации для поиска по базе знаний."""

    query: str


@tool(args_schema=PasswordResetInput)
async def send_password_reset(email: str) -> dict:
    """Отправляет ссылку для сброса пароля на указанный email."""

    async def _action() -> dict:
        try:
            await send_reset_link_via_api(email)
        except APIError as exc:
            return {"success": False, "error": str(exc), "should_escalate": True}
        return {
            "success": True,
            "message": f"Ссылка для сброса пароля отправлена на {email}",
            "email": email,
        }

    result = await run_once("send_password_reset", {"email": email}, _action)
    _record(
        ToolCallRecord(
            tool="send_password_reset",
            success=bool(result.get("success")),
            should_escalate=bool(result.get("should_escalate")),
            error=result.get("error"),
            idempotent_replay=bool(result.get("idempotent_replay")),
        )
    )
    return result


@tool(args_schema=OrderStatusInput)
async def check_order_status(order_id: str) -> dict:
    """Проверяет статус заказа по его номеру."""
    try:
        status_data = await orders_api_breaker.call(lambda: fetch_order_status(order_id))
        result = {
            "success": True,
            "order_id": order_id,
            "status": status_data["status"],
            "estimated_delivery": status_data.get("estimated_delivery"),
            "message": f"Заказ {order_id} находится в статусе: {status_data['status']}",
        }
    except OrderNotFoundError:
        result = {
            "success": False,
            "error": f"Заказ {order_id} не найден",
            "should_escalate": True,
        }
    except Exception:
        # Недоступность или размыкание breaker-а: тикет уходит оператору,
        # а не подвисает в ожидании (spec §36).
        result = {
            "success": False,
            "error": "Система заказов временно недоступна",
            "should_escalate": True,
        }

    _record(
        ToolCallRecord(
            tool="check_order_status",
            success=bool(result.get("success")),
            should_escalate=bool(result.get("should_escalate")),
            error=result.get("error"),
        )
    )
    return result


@tool(args_schema=KnowledgeBaseInput)
async def search_knowledge_base(query: str) -> dict:
    """Ищет информацию в базе знаний компании."""
    documents = await search_kb(query, k=3)
    if not documents:
        result = {
            "success": False,
            "message": "Релевантная информация не найдена",
            "should_escalate": False,
        }
    else:
        result = {
            "success": True,
            "content": "\n".join(document.content for document in documents),
        }

    _record(
        ToolCallRecord(
            tool=KNOWLEDGE_BASE_TOOL,
            success=bool(result.get("success")),
            should_escalate=False,
        )
    )
    return result


TOOLS = [send_password_reset, check_order_status, search_knowledge_base]

SYSTEM_PROMPT = """Ты агент службы поддержки. Твоя задача — помочь клиенту решить проблему.

Используй доступные инструменты для решения запроса.
ВАЖНО: если инструмент вернул success=False и should_escalate=True, прекрати
попытки и объясни, что нужна помощь оператора.
Не утверждай, что действие выполнено, если соответствующий инструмент не был
вызван успешно. Если не хватает данных (email, номер заказа) — попроси их у
клиента, не придумывай значения.
Будь вежливым и конкретным в ответах.

Текущий запрос был классифицирован как: {intent}"""


@lru_cache
def _get_agent(intent: Intent):
    """Граф агента компилируется один раз на намерение, а не на каждый тикет."""
    return create_agent(
        get_llm(),
        TOOLS,
        system_prompt=SYSTEM_PROMPT.format(intent=intent.value),
    )


def _evaluate(
    intent: Intent, records: list[ToolCallRecord], response: str
) -> tuple[bool, bool]:
    """Возвращает (resolved, requires_escalation).

    Отсутствие сбоев не равно решению: тикет считается закрытым только если
    выполнено действие, которое его закрывает.
    """
    if any(record.should_escalate for record in records):
        return False, True

    required_tool = RESOLVING_TOOL_BY_INTENT.get(intent)
    if required_tool is not None:
        performed = any(
            record.tool == required_tool and record.success for record in records
        )
        return performed, not performed

    if intent in ADVISORY_INTENTS:
        informed = bool(response.strip()) and any(
            record.tool == KNOWLEDGE_BASE_TOOL and record.success for record in records
        )
        return informed, not informed

    # Для намерения нет автоматизированного действия — закрывать его нельзя.
    return False, True


async def auto_resolve(
    ticket_text: str,
    intent: Intent,
    deadline: Deadline | None = None,
) -> tuple[ResolutionAttempt, list[UsageRecord]]:
    settings = get_settings()
    deadline = deadline or Deadline.from_settings()
    records: list[ToolCallRecord] = []
    token = _tool_records.set(records)
    started = time.monotonic()
    try:
        agent = _get_agent(intent)

        async def _invoke() -> dict[str, Any]:
            # Цикл агента может содержать несколько обращений к модели, поэтому
            # шаг ограничен остатком бюджета тикета, а не одним таймаутом запроса.
            return await deadline.run(agent.ainvoke({"messages": [("user", ticket_text)]}))

        try:
            result = await llm_breaker.call(
                lambda: retry_async(
                    _invoke,
                    attempts=settings.llm_max_retries + 1,
                    deadline=deadline,
                    min_attempt_seconds=settings.llm_request_timeout_seconds,
                )
            )
        except Exception as exc:
            llm_errors_total.labels(step=STEP, error_type=type(exc).__name__).inc()
            raise

        messages = result.get("messages", [])
        last_ai = next(
            (
                message
                for message in reversed(messages)
                if getattr(message, "type", None) == "ai"
                and isinstance(getattr(message, "content", None), str)
                and message.content.strip()
            ),
            None,
        )
        response = last_ai.content if last_ai else ""

        resolved, requires_escalation = _evaluate(intent, records, response)
        usage = extract_usage(messages, step=STEP, model=settings.llm_model)

        return (
            ResolutionAttempt(
                resolved=resolved,
                requires_escalation=requires_escalation,
                response=response,
                tool_calls=list(records),
            ),
            usage,
        )
    finally:
        llm_request_duration.labels(step=STEP).observe(time.monotonic() - started)
        _tool_records.reset(token)
