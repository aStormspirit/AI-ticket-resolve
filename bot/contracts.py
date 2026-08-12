"""Единые контракты домена: статусы, очереди, причины эскалации, результат обработки.

Все слои (pipeline, policy engine, метрики, API) обязаны использовать эти типы,
чтобы наборы строковых литералов не расходились между модулями.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(str, Enum):
    AUTO_RESOLVED = "auto_resolved"
    ESCALATED = "escalated"
    ESCALATED_TOOL_FAILURE = "escalated_tool_failure"
    ESCALATED_AFTER_ATTEMPT = "escalated_after_attempt"
    REQUIRES_MANUAL_HANDLING = "requires_manual_handling"


class Queue(str, Enum):
    AUTO_RESOLVE = "auto_resolve_queue"
    GENERAL_SUPPORT = "general_support_queue"
    MANUAL_REVIEW = "manual_review_queue"
    ESCALATION = "escalation_queue"
    REFUNDS = "refunds_queue"


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    SENSITIVE_TOPIC = "sensitive_topic"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    EXPLICIT_REQUEST = "explicit_request"
    BUSINESS_RULE = "business_rule"
    TOOL_FAILURE = "tool_failure"
    UNRESOLVED = "unresolved"
    TIMEOUT = "timeout"
    LLM_UNAVAILABLE = "llm_unavailable"
    INTERNAL_ERROR = "internal_error"


class Intent(str, Enum):
    PASSWORD_RESET = "password_reset"
    ORDER_STATUS = "order_status"
    ADDRESS_CHANGE = "address_change"
    REFUND_REQUEST = "refund_request"
    COMPLAINT = "complaint"
    GENERAL_INQUIRY = "general_inquiry"

    @classmethod
    def coerce(cls, value: str) -> "Intent":
        """Модель может вернуть категорию не из списка — приводим к общей."""
        try:
            return cls(value)
        except ValueError:
            return cls.GENERAL_INQUIRY


INTENT_DESCRIPTIONS: dict[Intent, str] = {
    Intent.PASSWORD_RESET: "Сброс или восстановление пароля",
    Intent.ORDER_STATUS: "Проверка статуса заказа",
    Intent.ADDRESS_CHANGE: "Изменение адреса доставки",
    Intent.REFUND_REQUEST: "Запрос возврата средств",
    Intent.COMPLAINT: "Жалоба на качество товара или обслуживания",
    Intent.GENERAL_INQUIRY: "Общий вопрос или другое",
}

# Инструмент, успешный вызов которого считается фактическим решением тикета.
# Ответ без вызова такого инструмента решением не является.
RESOLVING_TOOL_BY_INTENT: dict[Intent, str] = {
    Intent.PASSWORD_RESET: "send_password_reset",
    Intent.ORDER_STATUS: "check_order_status",
}

# Намерения, которые закрываются консультацией по базе знаний, без действия.
ADVISORY_INTENTS: frozenset[Intent] = frozenset({Intent.GENERAL_INQUIRY})


class ToolCallRecord(BaseModel):
    """Структурный след вызова инструмента, а не парсинг текста сообщения."""

    tool: str
    success: bool
    should_escalate: bool = False
    error: str | None = None
    idempotent_replay: bool = False


class UsageRecord(BaseModel):
    """Потребление токенов на одном шаге пайплайна."""

    step: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CostBreakdown(BaseModel):
    llm_api: float = 0.0
    infrastructure: float = 0.0
    total: float = 0.0


class ResolutionAttempt(BaseModel):
    """Результат попытки автоматического разрешения."""

    resolved: bool = False
    requires_escalation: bool = False
    response: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    error: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [call.tool for call in self.tool_calls]

    @property
    def failed_tools(self) -> list[str]:
        return [call.tool for call in self.tool_calls if not call.success]


class TicketResult(BaseModel):
    """Единственная форма ответа пайплайна."""

    ticket_id: str
    status: TicketStatus
    intent: Intent
    priority: Priority
    assigned_to: Queue
    reason: EscalationReason | None = None
    reasoning: str = ""
    confidence: float | None = None
    resolution: ResolutionAttempt | None = None
    handle_time_seconds: float | None = None
    usage: list[UsageRecord] = Field(default_factory=list)
    cost: CostBreakdown | None = None
    degraded: bool = False

    @property
    def is_escalated(self) -> bool:
        return self.status is not TicketStatus.AUTO_RESOLVED

    @property
    def resolution_type(self) -> str:
        return "auto" if self.status is TicketStatus.AUTO_RESOLVED else "manual"
