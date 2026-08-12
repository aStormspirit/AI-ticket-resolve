from __future__ import annotations

import pytest

from bot.idempotency import build_key, reset_scope, run_once, set_scope

pytestmark = pytest.mark.asyncio


async def test_action_runs_once_per_scope() -> None:
    calls = {"count": 0}

    async def action() -> dict:
        calls["count"] += 1
        return {"success": True, "message": "отправлено"}

    token = set_scope("T-1")
    try:
        first = await run_once("send_password_reset", {"email": "a@b.com"}, action)
        second = await run_once("send_password_reset", {"email": "a@b.com"}, action)
    finally:
        reset_scope(token)

    assert calls["count"] == 1
    assert first.get("idempotent_replay") is None
    assert second["idempotent_replay"] is True


async def test_different_tickets_do_not_share_keys() -> None:
    calls = {"count": 0}

    async def action() -> dict:
        calls["count"] += 1
        return {"success": True}

    for ticket_id in ("T-1", "T-2"):
        token = set_scope(ticket_id)
        try:
            await run_once("send_password_reset", {"email": "a@b.com"}, action)
        finally:
            reset_scope(token)

    assert calls["count"] == 2


async def test_failed_action_is_retried() -> None:
    calls = {"count": 0}

    async def failing() -> dict:
        calls["count"] += 1
        return {"success": False, "error": "api down"}

    token = set_scope("T-3")
    try:
        await run_once("send_password_reset", {"email": "a@b.com"}, failing)
        await run_once("send_password_reset", {"email": "a@b.com"}, failing)
    finally:
        reset_scope(token)

    assert calls["count"] == 2


async def test_keys_depend_on_arguments() -> None:
    token = set_scope("T-4")
    try:
        first = build_key("send_password_reset", email="a@b.com")
        second = build_key("send_password_reset", email="c@d.com")
    finally:
        reset_scope(token)

    assert first != second


async def test_local_fallback_when_redis_unavailable(fake_redis) -> None:
    fake_redis.fail = True
    calls = {"count": 0}

    async def action() -> dict:
        calls["count"] += 1
        return {"success": True}

    token = set_scope("T-5")
    try:
        await run_once("send_password_reset", {"email": "a@b.com"}, action)
        result = await run_once("send_password_reset", {"email": "a@b.com"}, action)
    finally:
        reset_scope(token)

    assert calls["count"] == 1
    assert result["idempotent_replay"] is True
