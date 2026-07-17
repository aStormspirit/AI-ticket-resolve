from enum import Enum
from dataclasses import dataclass

from .classifier import IntentClassification


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    SENSITIVE_TOPIC = "sensitive_topic"
    NEGATIVE_SENTIMENT = "negative_sentiment"
    EXPLICIT_REQUEST = "explicit_request"
    BUSINESS_RULE = "business_rule"


@dataclass
class ProcessingDecision:
    priority: Priority
    should_escalate: bool
    escalation_reason: EscalationReason | None
    assigned_queue: str
    can_use_tools: bool
    reasoning: str


class PolicyEngine:
    """Централизованный движок для принятия решений по обработке тикетов"""

    # Пороги уверенности калибруются на размеченной выборке
    CONFIDENCE_THRESHOLD = 0.7

    # Намерения, требующие человеческого участия
    SENSITIVE_INTENTS = {"complaint", "refund_request"}

    # Ключевые слова для детекции негатива
    NEGATIVE_KEYWORDS = ["ужасно", "отвратительно", "возмутительно", "жалоба"]

    # Фразы явного запроса оператора
    HUMAN_REQUEST_PHRASES = ["с оператором", "с человеком", "с менеджером"]

    def decide(
        self, intent: IntentClassification, ticket_text: str
    ) -> ProcessingDecision:
        """Единая точка принятия решения об обработке тикета"""

        # Правило 1: Явный запрос клиента
        if self._is_human_requested(ticket_text):
            return ProcessingDecision(
                priority=Priority.MEDIUM,
                should_escalate=True,
                escalation_reason=EscalationReason.EXPLICIT_REQUEST,
                assigned_queue="general_support_queue",
                can_use_tools=False,
                reasoning="Клиент явно попросил общения с человеком",
            )

        # Правило 2: Низкая уверенность модели
        if intent.confidence < self.CONFIDENCE_THRESHOLD:
            return ProcessingDecision(
                priority=Priority.MEDIUM,
                should_escalate=True,
                escalation_reason=EscalationReason.LOW_CONFIDENCE,
                assigned_queue="manual_review_queue",
                can_use_tools=False,
                reasoning=f"Уверенность {intent.confidence:.2f} ниже порога {self.CONFIDENCE_THRESHOLD}",
            )

        # Правило 3: Чувствительные темы
        if intent.intent in self.SENSITIVE_INTENTS:
            return ProcessingDecision(
                priority=Priority.HIGH,
                should_escalate=True,
                escalation_reason=EscalationReason.SENSITIVE_TOPIC,
                assigned_queue=(
                    "escalation_queue"
                    if intent.intent == "complaint"
                    else "refunds_queue"
                ),
                can_use_tools=False,
                reasoning=f"Намерение {intent.intent} требует персонального внимания",
            )

        # Правило 4: Негативная тональность
        if self._has_negative_sentiment(ticket_text):
            return ProcessingDecision(
                priority=Priority.HIGH,
                should_escalate=True,
                escalation_reason=EscalationReason.NEGATIVE_SENTIMENT,
                assigned_queue="escalation_queue",
                can_use_tools=False,
                reasoning="Обнаружена негативная тональность в тексте",
            )

        # Правило 5: Автоматическая обработка разрешена
        return ProcessingDecision(
            priority=(
                Priority.LOW
                if intent.intent in ["password_reset", "order_status"]
                else Priority.MEDIUM
            ),
            should_escalate=False,
            escalation_reason=None,
            assigned_queue="auto_resolve_queue",
            can_use_tools=True,
            reasoning=f"Намерение {intent.intent} подходит для автоматизации",
        )

    def _is_human_requested(self, text: str) -> bool:
        return any(phrase in text.lower() for phrase in self.HUMAN_REQUEST_PHRASES)

    def _has_negative_sentiment(self, text: str) -> bool:
        return any(word in text.lower() for word in self.NEGATIVE_KEYWORDS)


# Инициализация единого policy engine
policy_engine = PolicyEngine()
