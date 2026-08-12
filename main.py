"""CLI-демонстрация пайплайна: обработка тикета без запуска сервиса."""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

from bot.config import get_settings
from bot.pipeline import process_ticket
from bot.storage import close_redis
from observability.logging import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_TICKET = "У меня медленно грузится сайт"


async def _run(ticket_text: str, ticket_id: str) -> int:
    try:
        result = await process_ticket(ticket_text, ticket_id)
    finally:
        await close_redis()

    print("\n=== Результат обработки тикета ===")
    print(result.model_dump_json(indent=2, exclude_none=True))

    if result.resolution and result.resolution.response:
        print("\n=== Ответ агента ===")
        print(result.resolution.response)

    return 0 if not result.degraded else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Обработка тикета службы поддержки")
    parser.add_argument("text", nargs="?", default=DEFAULT_TICKET, help="Текст обращения")
    parser.add_argument(
        "--ticket-id",
        default=None,
        help="Идентификатор тикета (по умолчанию генерируется)",
    )
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    ticket_id = args.ticket_id or f"T-{uuid.uuid4().hex[:8]}"

    return asyncio.run(_run(args.text, ticket_id))


if __name__ == "__main__":
    raise SystemExit(main())
