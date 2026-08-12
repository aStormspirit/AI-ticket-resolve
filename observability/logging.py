"""Настройка логирования: correlation id через contextvars и PII-фильтр.

ticket_id не протаскивается вручную через сигнатуры — он живёт в contextvar и
подставляется в каждую запись лога фильтром.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .pii import scrub

_ticket_id: ContextVar[str] = ContextVar("ticket_id", default="-")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [ticket=%(ticket_id)s]: %(message)s"
COLOR_LOG_FORMAT = (
    "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(name)s "
    "[ticket=%(ticket_id)s]: %(message)s"
)


@contextmanager
def ticket_context(ticket_id: str) -> Iterator[None]:
    """Привязывает correlation id ко всем логам внутри блока."""
    token = _ticket_id.set(ticket_id)
    try:
        yield
    finally:
        _ticket_id.reset(token)


def current_ticket_id() -> str:
    return _ticket_id.get()


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ticket_id = _ticket_id.get()
        return True


class PIIFilter(logging.Filter):
    """Маскирует PII в сообщении и аргументах до форматирования записи."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            # Само сообщение может содержать %-плейсхолдеры, PII приходит в args,
            # но текст тоже маскируем на случай f-строк.
            record.msg = scrub(record.msg)
        if record.args:
            record.args = scrub(record.args)
        return True


def configure_logging(level: str = "INFO", *, use_colors: bool | None = None) -> None:
    """Идемпотентно настраивает корневой логгер."""
    if use_colors is None:
        use_colors = sys.stderr.isatty()

    handler = logging.StreamHandler(sys.stderr)
    if use_colors:
        try:
            from colorlog import ColoredFormatter

            handler.setFormatter(ColoredFormatter(COLOR_LOG_FORMAT))
        except ImportError:
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
    else:
        handler.setFormatter(logging.Formatter(LOG_FORMAT))

    handler.addFilter(CorrelationIdFilter())
    handler.addFilter(PIIFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
