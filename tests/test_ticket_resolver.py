from __future__ import annotations

import pytest

from bot.contracts import Intent, ToolCallRecord
from bot.idempotency import set_scope
from bot.ticket_resolver import (
    _evaluate,
    _tool_records,
    check_order_status,
    search_knowledge_base,
    send_password_reset,
)


def test_resolution_requires_the_resolving_tool() -> None:
    resolved, escalate = _evaluate(Intent.PASSWORD_RESET, [], "Я вам помогу")

    assert resolved is False
    assert escalate is True


def test_resolution_accepted_when_action_performed() -> None:
    records = [ToolCallRecord(tool="send_password_reset", success=True)]

    resolved, escalate = _evaluate(Intent.PASSWORD_RESET, records, "Готово")

    assert resolved is True
    assert escalate is False


def test_tool_failure_forces_escalation() -> None:
    records = [
        ToolCallRecord(tool="check_order_status", success=False, should_escalate=True)
    ]

    resolved, escalate = _evaluate(Intent.ORDER_STATUS, records, "Нужен оператор")

    assert resolved is False
    assert escalate is True


def test_advisory_intent_resolved_via_knowledge_base() -> None:
    records = [ToolCallRecord(tool="search_knowledge_base", success=True)]

    resolved, escalate = _evaluate(Intent.GENERAL_INQUIRY, records, "Вот инструкция")

    assert resolved is True
    assert escalate is False


def test_intent_without_automation_is_escalated() -> None:
    records = [ToolCallRecord(tool="search_knowledge_base", success=True)]

    resolved, escalate = _evaluate(Intent.ADDRESS_CHANGE, records, "Вот инструкция")

    assert resolved is False
    assert escalate is True


async def test_password_reset_tool_is_idempotent() -> None:
    token = set_scope("T-idem")
    records: list[ToolCallRecord] = []
    records_token = _tool_records.set(records)
    try:
        first = await send_password_reset.ainvoke({"email": "user@example.com"})
        second = await send_password_reset.ainvoke({"email": "user@example.com"})
    finally:
        _tool_records.reset(records_token)
        from bot.idempotency import reset_scope

        reset_scope(token)

    assert first["success"] is True
    assert second["success"] is True
    assert second["idempotent_replay"] is True
    assert records[1].idempotent_replay is True


async def test_unknown_order_escalates() -> None:
    records: list[ToolCallRecord] = []
    token = _tool_records.set(records)
    try:
        result = await check_order_status.ainvoke({"order_id": "ORD-000000"})
    finally:
        _tool_records.reset(token)

    assert result["success"] is False
    assert result["should_escalate"] is True
    assert records[0].should_escalate is True


async def test_invalid_order_id_is_rejected_before_api_call() -> None:
    with pytest.raises(Exception):
        await check_order_status.ainvoke({"order_id": "12345"})


async def test_knowledge_base_search_returns_content() -> None:
    records: list[ToolCallRecord] = []
    token = _tool_records.set(records)
    try:
        result = await search_knowledge_base.ainvoke({"query": "как сбросить пароль"})
    finally:
        _tool_records.reset(token)

    assert result["success"] is True
    assert "пароль" in result["content"].lower()


async def test_knowledge_base_degrades_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    from bot import knowledge_base

    def _broken(*args, **kwargs):
        raise knowledge_base.KnowledgeBaseUnavailable("хранилище недоступно")

    monkeypatch.setattr(
        knowledge_base.KnowledgeBase, "similarity_search", _broken, raising=True
    )

    records: list[ToolCallRecord] = []
    token = _tool_records.set(records)
    try:
        result = await search_knowledge_base.ainvoke({"query": "пароль"})
    finally:
        _tool_records.reset(token)

    assert result["success"] is False
    assert result["should_escalate"] is False
