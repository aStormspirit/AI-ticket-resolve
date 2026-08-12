"""Маскирование персональных данных в логах и трейсах (spec §33).

Тикеты содержат email, телефоны, адреса и номера карт. Сырой текст тикета и
reasoning модели (которая охотно цитирует обращение) не должны попадать в
логи, метрики или внешние трейс-системы в открытом виде.
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+7|8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")

MASK_EMAIL = "<email>"
MASK_PHONE = "<phone>"
MASK_CARD = "<card>"
MASK_IP = "<ip>"

_MAX_TEXT_LENGTH = 500


def mask_email(value: str) -> str:
    """Сохраняет первую букву и домен: v***@example.com — для поддержки диагностики."""

    def _replace(match: re.Match[str]) -> str:
        local, _, domain = match.group(0).partition("@")
        head = local[0] if local else ""
        return f"{head}***@{domain}"

    return _EMAIL.sub(_replace, value)


def mask_pii(value: str) -> str:
    """Заменяет все распознанные PII-паттерны в строке."""
    masked = mask_email(value)
    masked = _PHONE.sub(MASK_PHONE, masked)
    masked = _CARD.sub(MASK_CARD, masked)
    masked = _IPV4.sub(MASK_IP, masked)
    return masked


def scrub(value: Any) -> Any:
    """Рекурсивно маскирует PII в произвольной структуре."""
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        masked = [scrub(item) for item in value]
        return type(value)(masked)
    return value


def safe_preview(text: str, limit: int = _MAX_TEXT_LENGTH) -> str:
    """Маскированный и укороченный фрагмент текста для логов."""
    masked = mask_pii(text)
    if len(masked) <= limit:
        return masked
    return f"{masked[:limit]}...[truncated {len(masked) - limit} chars]"
