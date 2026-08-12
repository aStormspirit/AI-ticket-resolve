from __future__ import annotations

import logging

from observability.logging import PIIFilter, configure_logging, ticket_context
from observability.pii import mask_email, mask_pii, safe_preview, scrub


def test_email_keeps_domain_but_hides_local_part() -> None:
    assert mask_email("ivan.petrov@example.com") == "i***@example.com"


def test_phone_and_card_are_masked() -> None:
    masked = mask_pii("Телефон +7 (999) 123-45-67, карта 4111 1111 1111 1111")

    assert "999" not in masked
    assert "4111" not in masked
    assert "<phone>" in masked
    assert "<card>" in masked


def test_scrub_walks_nested_structures() -> None:
    payload = {"user": {"email": "a@b.com"}, "contacts": ["c@d.com"]}

    assert scrub(payload) == {"user": {"email": "a***@b.com"}, "contacts": ["c***@d.com"]}


def test_safe_preview_truncates() -> None:
    preview = safe_preview("а" * 600, limit=100)

    assert preview.startswith("а" * 100)
    assert "truncated" in preview


def test_log_filter_masks_arguments(caplog) -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Клиент %s",
        args=("secret.user@mail.ru",),
        exc_info=None,
    )

    PIIFilter().filter(record)

    assert record.getMessage() == "Клиент s***@mail.ru"


def test_correlation_id_is_attached(capsys) -> None:
    configure_logging("INFO", use_colors=False)
    with ticket_context("T-123"):
        logging.getLogger("test").info("сообщение")

    assert "ticket=T-123" in capsys.readouterr().err
