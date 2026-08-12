from __future__ import annotations

from types import SimpleNamespace

from bot.contracts import (
    Intent,
    Priority,
    Queue,
    ResolutionAttempt,
    TicketResult,
    TicketStatus,
    ToolCallRecord,
    UsageRecord,
)
from observability.cost import (
    build_cost_breakdown,
    calculate_infrastructure_cost_per_ticket,
    calculate_llm_cost_rubles,
    extract_usage,
)
from observability.metrics import (
    REGISTRY,
    tickets_auto_resolved_total,
    tickets_escalated_total,
    track_ticket_result,
)


def test_extract_usage_reads_usage_metadata() -> None:
    messages = [
        SimpleNamespace(usage_metadata={"input_tokens": 120, "output_tokens": 30}),
        SimpleNamespace(usage_metadata=None),
        SimpleNamespace(),
    ]

    usage = extract_usage(messages, step="classification", model="openai/gpt-5")

    assert len(usage) == 1
    assert usage[0].total_tokens == 150


def test_llm_cost_uses_price_table() -> None:
    usage = [
        UsageRecord(
            step="classification",
            model="openai/gpt-5",
            input_tokens=1_000_000,
            output_tokens=0,
        )
    ]

    # 1M входных токенов * $1.25 * 90 ₽/$ = 112.5 ₽
    assert calculate_llm_cost_rubles(usage) == 112.5


def test_infrastructure_cost_is_amortized_per_ticket() -> None:
    assert calculate_infrastructure_cost_per_ticket() == 1.8


def test_cost_breakdown_sums_components() -> None:
    usage = [
        UsageRecord(
            step="resolution",
            model="openai/gpt-5",
            input_tokens=10_000,
            output_tokens=1_000,
        )
    ]

    breakdown = build_cost_breakdown(usage)

    assert breakdown.total == round(breakdown.llm_api + breakdown.infrastructure, 4)
    assert breakdown.infrastructure == 1.8


def test_track_ticket_result_updates_counters() -> None:
    before = (
        REGISTRY.get_sample_value(
            "support_tickets_auto_resolved_total", {"intent": "password_reset"}
        )
        or 0.0
    )

    track_ticket_result(
        TicketResult(
            ticket_id="T-metrics",
            status=TicketStatus.AUTO_RESOLVED,
            intent=Intent.PASSWORD_RESET,
            priority=Priority.LOW,
            assigned_to=Queue.AUTO_RESOLVE,
            handle_time_seconds=1.5,
            resolution=ResolutionAttempt(
                resolved=True,
                tool_calls=[ToolCallRecord(tool="send_password_reset", success=True)],
            ),
        )
    )

    after = REGISTRY.get_sample_value(
        "support_tickets_auto_resolved_total", {"intent": "password_reset"}
    )
    assert after == before + 1

    tool_calls = REGISTRY.get_sample_value(
        "support_tool_calls_total",
        {"tool": "send_password_reset", "outcome": "success"},
    )
    assert tool_calls is not None and tool_calls >= 1


def test_escalation_counter_uses_reason_label() -> None:
    from bot.contracts import EscalationReason

    track_ticket_result(
        TicketResult(
            ticket_id="T-escalated",
            status=TicketStatus.ESCALATED,
            intent=Intent.REFUND_REQUEST,
            priority=Priority.HIGH,
            assigned_to=Queue.REFUNDS,
            reason=EscalationReason.SENSITIVE_TOPIC,
            handle_time_seconds=0.5,
        )
    )

    value = REGISTRY.get_sample_value(
        "support_tickets_escalated_total",
        {"reason": "sensitive_topic", "intent": "refund_request"},
    )
    assert value is not None and value >= 1
    assert tickets_auto_resolved_total is not tickets_escalated_total
