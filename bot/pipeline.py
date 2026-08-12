"""Оркестрация обработки тикета.

Intent classification → auto-triage → auto-resolve → HITL escalation.
Любой отказ по пути ведёт к эскалации оператору, а не к исключению наружу:
недоступность модели не должна оставлять обращение без обработки (spec §32).
"""

from __future__ import annotations

import logging
import time

from observability.cost import build_cost_breakdown
from observability.logging import ticket_context
from observability.metrics import track_ticket_received, track_ticket_result
from observability.pii import safe_preview

from .classifier import classify_intent
from .contracts import (
    EscalationReason,
    Intent,
    Queue,
    ResolutionAttempt,
    TicketResult,
    TicketStatus,
    UsageRecord,
)
from .idempotency import reset_scope, set_scope
from .policy_engine import ProcessingDecision, policy_engine
from .resilience import CircuitBreakerOpen, Deadline, DeadlineExceeded, StepTimeout
from .ticket_resolver import auto_resolve

logger = logging.getLogger(__name__)


async def process_ticket(
    ticket_text: str,
    ticket_id: str,
    deadline: Deadline | None = None,
) -> TicketResult:
    """Полный цикл обработки одного обращения."""
    started = time.monotonic()
    deadline = deadline or Deadline.from_settings()
    usage: list[UsageRecord] = []

    scope_token = set_scope(ticket_id)
    try:
        with ticket_context(ticket_id):
            logger.info("Получен тикет: %s", safe_preview(ticket_text))

            # Шаг 1: классификация намерения
            try:
                classification_result = await classify_intent(ticket_text, deadline)
            except Exception as exc:
                result = _degraded_result(
                    ticket_id=ticket_id,
                    exc=exc,
                    intent=Intent.GENERAL_INQUIRY,
                    usage=usage,
                )
                return _finalize(result, started)

            classification = classification_result.classification
            usage.extend(classification_result.usage)
            intent = Intent.coerce(classification.intent)

            logger.info(
                "Классифицировано: intent=%s confidence=%.2f reasoning=%s",
                intent.value,
                classification.confidence,
                safe_preview(classification.reasoning, limit=200),
            )

            # Шаг 2: решение policy engine
            decision = policy_engine.decide(
                intent, classification.confidence, ticket_text
            )
            logger.info(
                "Маршрутизация: queue=%s priority=%s escalate=%s tools_allowed=%s reason=%s",
                decision.assigned_queue.value,
                decision.priority.value,
                decision.should_escalate,
                decision.can_use_tools,
                decision.reasoning,
            )

            # Шаг 3: немедленная эскалация
            if decision.should_escalate or not decision.can_use_tools:
                result = TicketResult(
                    ticket_id=ticket_id,
                    status=TicketStatus.ESCALATED,
                    intent=intent,
                    priority=decision.priority,
                    assigned_to=decision.assigned_queue,
                    reason=decision.escalation_reason or EscalationReason.BUSINESS_RULE,
                    reasoning=decision.reasoning,
                    confidence=classification.confidence,
                    usage=usage,
                )
                return _finalize(result, started)

            # Шаг 4: попытка автоматического разрешения
            logger.info("Запуск автоматического разрешения, intent=%s", intent.value)
            try:
                resolution, resolution_usage = await auto_resolve(
                    ticket_text, intent, deadline
                )
                usage.extend(resolution_usage)
            except Exception as exc:
                result = _degraded_result(
                    ticket_id=ticket_id,
                    exc=exc,
                    intent=intent,
                    usage=usage,
                    confidence=classification.confidence,
                )
                return _finalize(result, started)

            logger.info(
                "Автоматическое разрешение: resolved=%s escalate=%s tools=%s",
                resolution.resolved,
                resolution.requires_escalation,
                resolution.tools_used,
            )

            # Шаг 5: интерпретация результата
            result = _result_from_resolution(
                ticket_id=ticket_id,
                intent=intent,
                decision=decision,
                resolution=resolution,
                confidence=classification.confidence,
                usage=usage,
            )
            return _finalize(result, started)
    finally:
        reset_scope(scope_token)


def _result_from_resolution(
    *,
    ticket_id: str,
    intent: Intent,
    decision: ProcessingDecision,
    resolution: ResolutionAttempt,
    confidence: float,
    usage: list[UsageRecord],
) -> TicketResult:
    if resolution.resolved and not resolution.requires_escalation:
        return TicketResult(
            ticket_id=ticket_id,
            status=TicketStatus.AUTO_RESOLVED,
            intent=intent,
            priority=decision.priority,
            assigned_to=decision.assigned_queue,
            reasoning=decision.reasoning,
            confidence=confidence,
            resolution=resolution,
            usage=usage,
        )

    tool_failed = any(call.should_escalate for call in resolution.tool_calls)
    if tool_failed:
        status = TicketStatus.ESCALATED_TOOL_FAILURE
        reason = EscalationReason.TOOL_FAILURE
    else:
        status = TicketStatus.ESCALATED_AFTER_ATTEMPT
        reason = EscalationReason.UNRESOLVED

    return TicketResult(
        ticket_id=ticket_id,
        status=status,
        intent=intent,
        priority=decision.priority,
        assigned_to=Queue.MANUAL_REVIEW,
        reason=reason,
        reasoning=(
            "Инструмент вернул ошибку, требуется оператор"
            if tool_failed
            else "Автоматическое действие не выполнено, требуется оператор"
        ),
        confidence=confidence,
        resolution=resolution,
        usage=usage,
    )


def _degraded_result(
    *,
    ticket_id: str,
    exc: BaseException,
    intent: Intent,
    usage: list[UsageRecord],
    confidence: float | None = None,
) -> TicketResult:
    """Единый путь деградации: тикет уходит оператору с указанием причины."""
    if isinstance(exc, (DeadlineExceeded, StepTimeout)):
        reason = EscalationReason.TIMEOUT
    elif isinstance(exc, CircuitBreakerOpen):
        reason = EscalationReason.LLM_UNAVAILABLE
    else:
        reason = EscalationReason.INTERNAL_ERROR

    logger.error(
        "Обработка деградирована (%s): %s",
        reason.value,
        exc,
        exc_info=reason is EscalationReason.INTERNAL_ERROR,
    )

    decision = policy_engine.degraded_decision(reason)
    return TicketResult(
        ticket_id=ticket_id,
        status=TicketStatus.REQUIRES_MANUAL_HANDLING,
        intent=intent,
        priority=decision.priority,
        assigned_to=decision.assigned_queue,
        reason=reason,
        reasoning=decision.reasoning,
        confidence=confidence,
        usage=usage,
        degraded=True,
    )


def _finalize(result: TicketResult, started: float) -> TicketResult:
    result.handle_time_seconds = round(time.monotonic() - started, 3)
    result.cost = build_cost_breakdown(result.usage)
    # Учитываем тикет здесь, а не после классификации: иначе обращения, упавшие
    # на первом шаге, не попадают в знаменатель auto-resolution rate.
    track_ticket_received(result.intent.value)
    track_ticket_result(result)
    logger.info(
        "Тикет завершён: status=%s queue=%s время=%.2fs стоимость=%.3f₽",
        result.status.value,
        result.assigned_to.value,
        result.handle_time_seconds,
        result.cost.total,
    )
    return result
