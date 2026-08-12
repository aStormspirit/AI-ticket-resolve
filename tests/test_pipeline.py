from __future__ import annotations

import pytest

from bot import pipeline as pipeline_module
from bot.classifier import ClassificationResult, IntentClassification
from bot.contracts import (
    EscalationReason,
    Intent,
    Queue,
    ResolutionAttempt,
    TicketStatus,
    ToolCallRecord,
    UsageRecord,
)
from bot.pipeline import process_ticket
from bot.resilience import CircuitBreakerOpen, DeadlineExceeded, StepTimeout

pytestmark = pytest.mark.asyncio


def _classification(
    intent: Intent = Intent.PASSWORD_RESET, confidence: float = 0.95
) -> ClassificationResult:
    return ClassificationResult(
        classification=IntentClassification(
            intent=intent, confidence=confidence, reasoning="тест"
        ),
        usage=[
            UsageRecord(
                step="classification",
                model="openai/gpt-5",
                input_tokens=100,
                output_tokens=20,
            )
        ],
    )


def _patch_classifier(monkeypatch: pytest.MonkeyPatch, result: ClassificationResult):
    async def _fake(ticket_text: str, deadline=None) -> ClassificationResult:
        return result

    monkeypatch.setattr(pipeline_module, "classify_intent", _fake)


def _patch_resolver(monkeypatch: pytest.MonkeyPatch, attempt: ResolutionAttempt):
    async def _fake(ticket_text: str, intent: Intent, deadline=None):
        return attempt, []

    monkeypatch.setattr(pipeline_module, "auto_resolve", _fake)


async def test_successful_auto_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_classifier(monkeypatch, _classification())
    _patch_resolver(
        monkeypatch,
        ResolutionAttempt(
            resolved=True,
            response="Ссылка отправлена",
            tool_calls=[ToolCallRecord(tool="send_password_reset", success=True)],
        ),
    )

    result = await process_ticket("Забыл пароль", "T-1")

    assert result.status is TicketStatus.AUTO_RESOLVED
    assert result.assigned_to is Queue.AUTO_RESOLVE
    assert result.handle_time_seconds is not None
    assert result.cost is not None and result.cost.total > 0


async def test_sensitive_intent_escalates_before_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_classifier(monkeypatch, _classification(Intent.REFUND_REQUEST))

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("auto_resolve не должен вызываться при эскалации")

    monkeypatch.setattr(pipeline_module, "auto_resolve", _must_not_run)

    result = await process_ticket("Верните деньги", "T-2")

    assert result.status is TicketStatus.ESCALATED
    assert result.reason is EscalationReason.SENSITIVE_TOPIC
    assert result.assigned_to is Queue.REFUNDS


async def test_tool_failure_is_escalated(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_classifier(monkeypatch, _classification(Intent.ORDER_STATUS))
    _patch_resolver(
        monkeypatch,
        ResolutionAttempt(
            resolved=False,
            requires_escalation=True,
            response="Нужен оператор",
            tool_calls=[
                ToolCallRecord(
                    tool="check_order_status",
                    success=False,
                    should_escalate=True,
                    error="не найден",
                )
            ],
        ),
    )

    result = await process_ticket("Где заказ ORD-000000", "T-3")

    assert result.status is TicketStatus.ESCALATED_TOOL_FAILURE
    assert result.reason is EscalationReason.TOOL_FAILURE
    assert result.assigned_to is Queue.MANUAL_REVIEW


async def test_chat_without_action_is_not_counted_as_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Агент ответил, но не выполнил действие — тикет закрывать нельзя."""
    _patch_classifier(monkeypatch, _classification(Intent.PASSWORD_RESET))
    _patch_resolver(
        monkeypatch,
        ResolutionAttempt(
            resolved=False,
            requires_escalation=True,
            response="Опишите проблему подробнее",
            tool_calls=[],
        ),
    )

    result = await process_ticket("Не могу войти", "T-4")

    assert result.status is TicketStatus.ESCALATED_AFTER_ATTEMPT
    assert result.reason is EscalationReason.UNRESOLVED


async def test_llm_unavailable_routes_to_operators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unavailable(ticket_text: str, deadline=None):
        raise CircuitBreakerOpen("llm")

    monkeypatch.setattr(pipeline_module, "classify_intent", _unavailable)

    result = await process_ticket("Любой текст", "T-5")

    assert result.status is TicketStatus.REQUIRES_MANUAL_HANDLING
    assert result.reason is EscalationReason.LLM_UNAVAILABLE
    assert result.degraded is True
    assert result.assigned_to is Queue.GENERAL_SUPPORT


async def test_deadline_exceeded_marks_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_classifier(monkeypatch, _classification())

    async def _too_slow(ticket_text: str, intent: Intent, deadline=None):
        raise DeadlineExceeded("бюджет исчерпан")

    monkeypatch.setattr(pipeline_module, "auto_resolve", _too_slow)

    result = await process_ticket("Забыл пароль", "T-6")

    assert result.status is TicketStatus.REQUIRES_MANUAL_HANDLING
    assert result.reason is EscalationReason.TIMEOUT
    assert result.degraded is True


async def test_step_timeout_is_reported_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_classifier(monkeypatch, _classification())

    async def _hanging(ticket_text: str, intent: Intent, deadline=None):
        raise StepTimeout("вызов завис")

    monkeypatch.setattr(pipeline_module, "auto_resolve", _hanging)

    result = await process_ticket("Забыл пароль", "T-8")

    assert result.reason is EscalationReason.TIMEOUT
    assert result.degraded is True


async def test_unexpected_error_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _broken(ticket_text: str, deadline=None):
        raise ValueError("невалидный ответ модели")

    monkeypatch.setattr(pipeline_module, "classify_intent", _broken)

    result = await process_ticket("Любой текст", "T-7")

    assert result.status is TicketStatus.REQUIRES_MANUAL_HANDLING
    assert result.reason is EscalationReason.INTERNAL_ERROR
