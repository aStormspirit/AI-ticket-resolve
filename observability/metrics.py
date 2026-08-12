"""Метрики Prometheus.

auto-resolution rate сознательно не считается в приложении: доля выводится
запросом rate() по счётчикам (см. docker/prometheus/rules.yml). Gauge,
усредняющий за всё время жизни процесса, вводит в заблуждение.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from bot.contracts import TicketResult, TicketStatus

REGISTRY = CollectorRegistry(auto_describe=True)

tickets_received_total = Counter(
    "support_tickets_received_total",
    "Общее количество полученных тикетов",
    ["intent"],
    registry=REGISTRY,
)

tickets_processed_total = Counter(
    "support_tickets_processed_total",
    "Тикеты, обработка которых завершилась, с разбивкой по итоговому статусу",
    ["status", "intent"],
    registry=REGISTRY,
)

tickets_auto_resolved_total = Counter(
    "support_tickets_auto_resolved_total",
    "Количество автоматически разрешённых тикетов",
    ["intent"],
    registry=REGISTRY,
)

tickets_escalated_total = Counter(
    "support_tickets_escalated_total",
    "Количество эскалированных тикетов",
    ["reason", "intent"],
    registry=REGISTRY,
)

ticket_handle_time = Histogram(
    "support_ticket_handle_time_seconds",
    "Время обработки тикета в секундах",
    ["resolution_type"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600],
    registry=REGISTRY,
)

ticket_cost = Histogram(
    "support_ticket_cost_rubles",
    "Стоимость обработки тикета в рублях",
    ["resolution_type", "cost_component"],
    buckets=[0.1, 0.5, 1, 5, 10, 50, 100],
    registry=REGISTRY,
)

llm_request_duration = Histogram(
    "support_llm_request_duration_seconds",
    "Длительность обращения к LLM по шагам пайплайна",
    ["step"],
    buckets=[0.25, 0.5, 1, 2, 3, 5, 8, 13, 21],
    registry=REGISTRY,
)

llm_tokens_total = Counter(
    "support_llm_tokens_total",
    "Потреблённые токены",
    ["step", "model", "kind"],
    registry=REGISTRY,
)

llm_errors_total = Counter(
    "support_llm_errors_total",
    "Ошибки обращения к LLM",
    ["step", "error_type"],
    registry=REGISTRY,
)

tool_calls_total = Counter(
    "support_tool_calls_total",
    "Вызовы инструментов агента",
    ["tool", "outcome"],
    registry=REGISTRY,
)

circuit_breaker_state = Gauge(
    "support_circuit_breaker_state",
    "Состояние circuit breaker: 0=closed, 1=half_open, 2=open",
    ["target"],
    registry=REGISTRY,
)

circuit_breaker_trips_total = Counter(
    "support_circuit_breaker_trips_total",
    "Количество размыканий circuit breaker",
    ["target"],
    registry=REGISTRY,
)

rate_limited_total = Counter(
    "support_rate_limited_total",
    "Количество отклонённых по rate limit запросов",
    ["scope"],
    registry=REGISTRY,
)

degraded_operations_total = Counter(
    "support_degraded_operations_total",
    "Работа в деградированном режиме: компонент недоступен, обработка продолжена",
    ["component"],
    registry=REGISTRY,
)

idempotent_replays_total = Counter(
    "support_idempotent_replays_total",
    "Повторные вызовы инструментов, отражённые ключом идемпотентности",
    ["tool"],
    registry=REGISTRY,
)

queue_depth = Gauge(
    "support_queue_depth",
    "Текущее количество задач в очереди обработки",
    ["queue"],
    registry=REGISTRY,
)

_ESCALATED_STATUSES = {
    TicketStatus.ESCALATED,
    TicketStatus.ESCALATED_TOOL_FAILURE,
    TicketStatus.ESCALATED_AFTER_ATTEMPT,
    TicketStatus.REQUIRES_MANUAL_HANDLING,
}


def track_ticket_received(intent: str) -> None:
    tickets_received_total.labels(intent=intent).inc()


def track_ticket_result(result: TicketResult) -> None:
    """Единая точка учёта завершённого тикета."""
    intent = result.intent.value
    status = result.status

    tickets_processed_total.labels(status=status.value, intent=intent).inc()

    if status is TicketStatus.AUTO_RESOLVED:
        tickets_auto_resolved_total.labels(intent=intent).inc()
    elif status in _ESCALATED_STATUSES:
        reason = result.reason.value if result.reason else "unknown"
        tickets_escalated_total.labels(reason=reason, intent=intent).inc()

    if result.handle_time_seconds is not None:
        ticket_handle_time.labels(resolution_type=result.resolution_type).observe(
            result.handle_time_seconds
        )

    if result.cost is not None:
        for component, value in (
            ("llm_api", result.cost.llm_api),
            ("infrastructure", result.cost.infrastructure),
            ("total", result.cost.total),
        ):
            ticket_cost.labels(
                resolution_type=result.resolution_type, cost_component=component
            ).observe(value)

    for record in result.usage:
        llm_tokens_total.labels(
            step=record.step, model=record.model, kind="input"
        ).inc(record.input_tokens)
        llm_tokens_total.labels(
            step=record.step, model=record.model, kind="output"
        ).inc(record.output_tokens)

    if result.resolution is not None:
        for call in result.resolution.tool_calls:
            outcome = "success" if call.success else "failure"
            tool_calls_total.labels(tool=call.tool, outcome=outcome).inc()
            if call.idempotent_replay:
                idempotent_replays_total.labels(tool=call.tool).inc()
