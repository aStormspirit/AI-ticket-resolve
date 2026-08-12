from __future__ import annotations

import pytest

from bot.contracts import EscalationReason, Intent, Priority, Queue
from bot.policy_engine import PolicyEngine


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine(confidence_threshold=0.7)


def test_simple_intent_goes_to_automation(engine: PolicyEngine) -> None:
    decision = engine.decide(Intent.PASSWORD_RESET, 0.95, "Забыл пароль от аккаунта")

    assert decision.should_escalate is False
    assert decision.can_use_tools is True
    assert decision.assigned_queue is Queue.AUTO_RESOLVE
    assert decision.priority is Priority.LOW


def test_low_confidence_goes_to_manual_review(engine: PolicyEngine) -> None:
    decision = engine.decide(Intent.GENERAL_INQUIRY, 0.5, "Непонятный запрос")

    assert decision.should_escalate is True
    assert decision.escalation_reason is EscalationReason.LOW_CONFIDENCE
    assert decision.assigned_queue is Queue.MANUAL_REVIEW
    assert decision.can_use_tools is False


def test_explicit_human_request_escalates(engine: PolicyEngine) -> None:
    decision = engine.decide(Intent.ORDER_STATUS, 0.99, "Хочу поговорить с оператором")

    assert decision.escalation_reason is EscalationReason.EXPLICIT_REQUEST
    assert decision.assigned_queue is Queue.GENERAL_SUPPORT


def test_refund_request_goes_to_refunds_queue(engine: PolicyEngine) -> None:
    decision = engine.decide(Intent.REFUND_REQUEST, 0.9, "Верните деньги за заказ")

    assert decision.escalation_reason is EscalationReason.SENSITIVE_TOPIC
    assert decision.assigned_queue is Queue.REFUNDS
    assert decision.priority is Priority.HIGH


def test_negative_sentiment_escalates(engine: PolicyEngine) -> None:
    decision = engine.decide(Intent.ORDER_STATUS, 0.9, "Это отвратительно, где заказ")

    assert decision.escalation_reason is EscalationReason.NEGATIVE_SENTIMENT
    assert decision.priority is Priority.HIGH


def test_priority_is_not_downgraded_by_earlier_rule(engine: PolicyEngine) -> None:
    """Жалоба с просьбой позвать оператора должна остаться HIGH."""
    decision = engine.decide(
        Intent.COMPLAINT, 0.95, "Хочу поговорить с оператором, это возмутительно"
    )

    assert decision.priority is Priority.HIGH
    assert decision.escalation_reason is EscalationReason.SENSITIVE_TOPIC


def test_threshold_comes_from_settings(settings) -> None:
    engine = PolicyEngine()

    assert engine.confidence_threshold == settings.confidence_threshold


def test_degraded_decision_routes_to_operators(engine: PolicyEngine) -> None:
    decision = engine.degraded_decision(EscalationReason.LLM_UNAVAILABLE)

    assert decision.should_escalate is True
    assert decision.can_use_tools is False
    assert decision.assigned_queue is Queue.GENERAL_SUPPORT
