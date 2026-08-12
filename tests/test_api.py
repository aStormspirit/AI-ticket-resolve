from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main as app_main
from app.main import create_app
from app.routers import tickets as tickets_router
from bot.contracts import Intent, Priority, Queue, TicketResult, TicketStatus
from bot.storage import TicketResultStore


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: dict[str, tuple[str, tuple[Any, ...]]] = {}

    async def enqueue_job(self, function: str, *args: Any, _job_id: str | None = None):
        job_id = _job_id or f"job:{len(self.jobs)}"
        if job_id in self.jobs:
            return None
        self.jobs[job_id] = (function, args)
        return object()

    async def aclose(self) -> None:
        return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    queue = FakeQueue()

    async def _no_active_job(*args: Any, **kwargs: Any) -> bool:
        return False

    async def _fake_pool() -> FakeQueue:
        return queue

    monkeypatch.setattr(tickets_router, "is_job_active", _no_active_job)
    monkeypatch.setattr(app_main, "create_queue_pool", _fake_pool)

    with TestClient(create_app()) as test_client:
        test_client.queue = queue  # type: ignore[attr-defined]
        yield test_client


def _sample_result(ticket_id: str) -> TicketResult:
    return TicketResult(
        ticket_id=ticket_id,
        status=TicketStatus.AUTO_RESOLVED,
        intent=Intent.PASSWORD_RESET,
        priority=Priority.LOW,
        assigned_to=Queue.AUTO_RESOLVE,
    )


def test_liveness_is_always_ok(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_dependencies(client: TestClient) -> None:
    response = client.get("/health/ready")

    body = response.json()
    assert body["redis"] is True
    assert body["llm_breaker"] == "closed"


def test_create_ticket_enqueues_job(client: TestClient) -> None:
    response = client.post("/api/v1/tickets", json={"text": "Забыл пароль"})

    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "queued"
    assert body["poll_url"] == f"/api/v1/tickets/{body['ticket_id']}"
    assert len(client.queue.jobs) == 1


def test_repeated_submission_does_not_duplicate_job(client: TestClient) -> None:
    payload = {"text": "Забыл пароль", "ticket_id": "T-100"}

    first = client.post("/api/v1/tickets", json=payload)
    second = client.post("/api/v1/tickets", json=payload)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["state"] == "already_queued"
    assert len(client.queue.jobs) == 1


def test_empty_text_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/tickets", json={"text": ""})

    assert response.status_code == 422


def test_get_unknown_ticket_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/tickets/T-unknown").status_code == 404


def test_get_ready_ticket_returns_result(client: TestClient) -> None:
    asyncio.run(TicketResultStore().save(_sample_result("T-200")))

    response = client.get("/api/v1/tickets/T-200")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["result"]["status"] == "auto_resolved"


def test_metrics_endpoint_exposes_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "support_tickets_received_total" in response.text


def test_demo_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Обработка тикетов поддержки" in response.text


def test_request_id_header_is_returned(client: TestClient) -> None:
    response = client.get("/health/live", headers={"X-Request-ID": "req-42"})

    assert response.headers["X-Request-ID"] == "req-42"
    assert "X-Response-Time" in response.headers
