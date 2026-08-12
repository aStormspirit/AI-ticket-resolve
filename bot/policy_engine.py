"""Детерминированный движок принятия решений.

Решение о маршрутизации принимает не LLM, а явные правила: это тестируемо,
воспроизводимо и не тратит бюджет латентности на лишний вызов модели.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import get_settings
from .contracts import EscalationReason, Intent, Priority, Queue

_PRIORITY_ORDER = {
    Priority.LOW: 0,
    Priority.MEDIUM: 1,
    Priority.HIGH: 2,
    Priority.CRITICAL: 3,
}


@dataclass(frozen=True)
class ProcessingDecision:
    priority: Priority
    should_escalate: bool
    escalation_reason: EscalationReason | None
    assigned_queue: Queue
    can_use_tools: bool
    reasoning: str


@dataclass(frozen=True)
class _Signal:
    """Сработавшее правило эскалации."""

    reason: EscalationReason
    priority: Priority
    queue: Queue
    explanation: str


class PolicyEngine:
    """Централизованный движок для принятия решений по обработке тикетов."""

    # Намерения, требующие человеческого участия
    SENSITIVE_INTENTS = frozenset({Intent.COMPLAINT, Intent.REFUND_REQUEST})

    # Намерения, для которых автоматическое решение приоритетно и дёшево
    LOW_PRIORITY_INTENTS = frozenset({Intent.PASSWORD_RESET, Intent.ORDER_STATUS})

    NEGATIVE_KEYWORDS = (
        "ужасно",
        "отвратительно",
        "возмутительно",
        "жалоба",
        "безобразие",
        "хамство",
    )

    HUMAN_REQUEST_PHRASES = (
        "с оператором",
        "с человеком",
        "с менеджером",
        "живой человек",
    )

    def __init__(self, confidence_threshold: float | None = None) -> None:
        # Порог живёт только здесь: дублирование его в промпте классификатора
        # приводило к тому, что неуверенные тикеты приходили как general_inquiry
        # с высокой confidence и уходили в автообработку.
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else get_settings().confidence_threshold
        )

    def decide(
        self, intent: Intent, confidence: float, ticket_text: str
    ) -> ProcessingDecision:
        """Единая точка принятия решения об обработке тикета."""
        signals = self._collect_signals(intent, confidence, ticket_text)

        if signals:
            # Причина и очередь — от правила с наивысшим приоритетом; сам приоритет
            # берём максимальный, чтобы жалоба с просьбой позвать оператора не
            # понижалась до MEDIUM.
            primary = max(signals, key=lambda signal: _PRIORITY_ORDER[signal.priority])
            priority = primary.priority
            return ProcessingDecision(
                priority=priority,
                should_escalate=True,
                escalation_reason=primary.reason,
                assigned_queue=primary.queue,
                can_use_tools=False,
                reasoning=primary.explanation,
            )

        return ProcessingDecision(
            priority=(
                Priority.LOW if intent in self.LOW_PRIORITY_INTENTS else Priority.MEDIUM
            ),
            should_escalate=False,
            escalation_reason=None,
            assigned_queue=Queue.AUTO_RESOLVE,
            can_use_tools=True,
            reasoning=f"Намерение {intent.value} подходит для автоматизации",
        )

    def degraded_decision(self, reason: EscalationReason) -> ProcessingDecision:
        """Решение при недоступности LLM или исчерпании бюджета времени (spec §32, §31)."""
        return ProcessingDecision(
            priority=Priority.MEDIUM,
            should_escalate=True,
            escalation_reason=reason,
            assigned_queue=Queue.GENERAL_SUPPORT,
            can_use_tools=False,
            reasoning=f"Автоматическая обработка недоступна: {reason.value}",
        )

    def _collect_signals(
        self, intent: Intent, confidence: float, ticket_text: str
    ) -> list[_Signal]:
        signals: list[_Signal] = []

        if self._is_human_requested(ticket_text):
            signals.append(
                _Signal(
                    reason=EscalationReason.EXPLICIT_REQUEST,
                    priority=Priority.MEDIUM,
                    queue=Queue.GENERAL_SUPPORT,
                    explanation="Клиент явно попросил общения с человеком",
                )
            )

        if confidence < self.confidence_threshold:
            signals.append(
                _Signal(
                    reason=EscalationReason.LOW_CONFIDENCE,
                    priority=Priority.MEDIUM,
                    queue=Queue.MANUAL_REVIEW,
                    explanation=(
                        f"Уверенность {confidence:.2f} ниже порога "
                        f"{self.confidence_threshold:.2f}"
                    ),
                )
            )

        if intent in self.SENSITIVE_INTENTS:
            signals.append(
                _Signal(
                    reason=EscalationReason.SENSITIVE_TOPIC,
                    priority=Priority.HIGH,
                    queue=(
                        Queue.ESCALATION
                        if intent is Intent.COMPLAINT
                        else Queue.REFUNDS
                    ),
                    explanation=f"Намерение {intent.value} требует персонального внимания",
                )
            )

        if self._has_negative_sentiment(ticket_text):
            signals.append(
                _Signal(
                    reason=EscalationReason.NEGATIVE_SENTIMENT,
                    priority=Priority.HIGH,
                    queue=Queue.ESCALATION,
                    explanation="Обнаружена негативная тональность в тексте",
                )
            )

        return signals

    def _is_human_requested(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in self.HUMAN_REQUEST_PHRASES)

    def _has_negative_sentiment(self, text: str) -> bool:
        lowered = text.lower()
        return any(word in lowered for word in self.NEGATIVE_KEYWORDS)


policy_engine = PolicyEngine()
