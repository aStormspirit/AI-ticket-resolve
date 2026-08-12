"""Схемы запросов и ответов HTTP API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from bot.contracts import TicketResult

MAX_TICKET_LENGTH = 8_000


class TicketCreateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TICKET_LENGTH, description="Текст обращения")
    ticket_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9_:-]+$",
        description="Внешний идентификатор; при повторной отправке обработка не дублируется",
    )


class AcceptedState(str, Enum):
    QUEUED = "queued"
    ALREADY_QUEUED = "already_queued"
    ALREADY_PROCESSED = "already_processed"


class TicketAcceptedResponse(BaseModel):
    ticket_id: str
    state: AcceptedState
    poll_url: str


class TicketStatusResponse(BaseModel):
    ticket_id: str
    ready: bool
    result: TicketResult | None = None


class HealthResponse(BaseModel):
    status: str
    redis: bool | None = None
    llm_breaker: str | None = None


class ErrorResponse(BaseModel):
    detail: str
