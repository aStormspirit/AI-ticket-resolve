"""Приём обращений и выдача результатов обработки."""

from __future__ import annotations

import logging
import uuid

from arq.connections import ArqRedis
from arq.jobs import Job, JobStatus
from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.deps import QueueDep, ResultStoreDep
from app.rate_limit import enforce_rate_limit
from app.schemas import (
    AcceptedState,
    TicketAcceptedResponse,
    TicketCreateRequest,
    TicketStatusResponse,
)
from observability.pii import safe_preview

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["tickets"],
    dependencies=[Depends(enforce_rate_limit)],
)

JOB_NAME = "process_ticket_job"
_ACTIVE_JOB_STATUSES = {JobStatus.deferred, JobStatus.queued, JobStatus.in_progress}


def job_id_for(ticket_id: str) -> str:
    return f"ticket:{ticket_id}"


async def is_job_active(queue: ArqRedis, ticket_id: str) -> bool:
    """Проверяет, обрабатывается ли тикет сейчас.

    Сбой Redis трактуется как отсутствие задачи: запрос статуса не должен
    возвращать 500 из-за недоступности вспомогательного хранилища.
    """
    try:
        status_value = await Job(job_id_for(ticket_id), queue).status()
    except Exception:
        logger.warning("Не удалось получить статус задачи для %s", ticket_id)
        return False
    return status_value in _ACTIVE_JOB_STATUSES


@router.post(
    "",
    response_model=TicketAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Принять обращение в обработку",
)
async def create_ticket(
    payload: TicketCreateRequest,
    queue: QueueDep,
    result_store: ResultStoreDep,
    response: Response,
) -> TicketAcceptedResponse:
    ticket_id = payload.ticket_id or f"T-{uuid.uuid4().hex[:12]}"
    poll_url = f"/api/v1/tickets/{ticket_id}"

    if await result_store.exists(ticket_id):
        response.status_code = status.HTTP_200_OK
        return TicketAcceptedResponse(
            ticket_id=ticket_id,
            state=AcceptedState.ALREADY_PROCESSED,
            poll_url=poll_url,
        )

    # _job_id делает постановку в очередь идемпотентной: повторная отправка того
    # же тикета не создаёт вторую задачу (spec §34).
    job = await queue.enqueue_job(
        JOB_NAME, payload.text, ticket_id, _job_id=job_id_for(ticket_id)
    )

    if job is None:
        response.status_code = status.HTTP_200_OK
        return TicketAcceptedResponse(
            ticket_id=ticket_id,
            state=AcceptedState.ALREADY_QUEUED,
            poll_url=poll_url,
        )

    logger.info(
        "Тикет %s поставлен в очередь: %s", ticket_id, safe_preview(payload.text, 120)
    )
    return TicketAcceptedResponse(
        ticket_id=ticket_id, state=AcceptedState.QUEUED, poll_url=poll_url
    )


@router.get(
    "/{ticket_id}",
    response_model=TicketStatusResponse,
    summary="Получить результат обработки",
)
async def get_ticket(
    ticket_id: str,
    queue: QueueDep,
    result_store: ResultStoreDep,
    response: Response,
) -> TicketStatusResponse:
    result = await result_store.get(ticket_id)
    if result is not None:
        return TicketStatusResponse(ticket_id=ticket_id, ready=True, result=result)

    if await is_job_active(queue, ticket_id):
        response.status_code = status.HTTP_202_ACCEPTED
        return TicketStatusResponse(ticket_id=ticket_id, ready=False)

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Тикет не найден"
    )
