"""Моки внешних интеграций.

Заменяются реальными клиентами без изменения кода инструментов: контракт —
async-функции, бросающие APIError/OrderNotFoundError.
"""

from __future__ import annotations

import asyncio
import logging

from observability.pii import mask_email

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Внешний сервис недоступен или вернул ошибку."""


class OrderNotFoundError(Exception):
    """Заказ с указанным номером не существует."""


# Номера, на которых демонстрируются ветки отказов.
UNKNOWN_ORDER_ID = "ORD-000000"
FAILING_ORDER_ID = "ORD-999999"


async def send_reset_link_via_api(email: str) -> dict:
    await asyncio.sleep(0)
    logger.info("Отправлена ссылка сброса пароля на %s", mask_email(email))
    return {"ok": True, "email": email}


async def fetch_order_status(order_id: str) -> dict:
    await asyncio.sleep(0)
    if order_id == UNKNOWN_ORDER_ID:
        raise OrderNotFoundError(order_id)
    if order_id == FAILING_ORDER_ID:
        raise APIError("orders service unavailable")

    logger.info("Запрошен статус заказа %s", order_id)
    return {
        "status": "shipped",
        "estimated_delivery": "2026-08-20",
    }
