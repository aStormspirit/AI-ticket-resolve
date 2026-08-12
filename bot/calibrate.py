"""Калибровка порога уверенности на размеченной выборке.

Порог подбирается так, чтобы автоматически обрабатывались только те обращения,
где классификатор действительно точен. Результат подставляется в
CONFIDENCE_THRESHOLD (переменная окружения), а не правится в коде.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sklearn.metrics import precision_recall_curve

from .classifier import classify_intent
from .contracts import Intent

logger = logging.getLogger(__name__)

DEFAULT_MIN_PRECISION = 0.9
CONSERVATIVE_THRESHOLD = 0.95


@dataclass
class CalibrationReport:
    threshold: float
    precision: float
    recall: float
    samples: int
    accuracy: float


async def calibrate_confidence_threshold(
    labeled_tickets: list[tuple[str, str]],
    min_precision: float = DEFAULT_MIN_PRECISION,
    concurrency: int = 4,
) -> CalibrationReport:
    """Подбирает порог с максимальным recall при заданной precision.

    Args:
        labeled_tickets: пары (текст тикета, истинное намерение)
        min_precision: минимально допустимая точность автообработки
        concurrency: количество параллельных обращений к модели
    """
    if not labeled_tickets:
        raise ValueError("Размеченная выборка пуста")

    semaphore = asyncio.Semaphore(concurrency)

    async def _classify(text: str, expected: str) -> tuple[float, bool]:
        async with semaphore:
            result = await classify_intent(text)
        predicted = Intent.coerce(result.classification.intent)
        return result.classification.confidence, predicted is Intent.coerce(expected)

    outcomes = await asyncio.gather(
        *(_classify(text, expected) for text, expected in labeled_tickets)
    )

    confidences = [confidence for confidence, _ in outcomes]
    correctness = [is_correct for _, is_correct in outcomes]
    accuracy = sum(correctness) / len(correctness)

    if len(set(correctness)) < 2:
        logger.warning(
            "В выборке только один класс исходов, кривая precision-recall не строится"
        )
        return CalibrationReport(
            threshold=CONSERVATIVE_THRESHOLD,
            precision=accuracy,
            recall=1.0 if all(correctness) else 0.0,
            samples=len(labeled_tickets),
            accuracy=accuracy,
        )

    precision, recall, thresholds = precision_recall_curve(correctness, confidences)

    # precision/recall длиннее thresholds на один элемент — последний соответствует
    # вырожденной точке (recall=0), поэтому срезаем его.
    candidates = [
        (thresholds[index], precision[index], recall[index])
        for index in range(len(thresholds))
        if precision[index] >= min_precision
    ]

    if not candidates:
        logger.warning(
            "Ни один порог не даёт precision >= %.2f, берём консервативный %.2f",
            min_precision,
            CONSERVATIVE_THRESHOLD,
        )
        return CalibrationReport(
            threshold=CONSERVATIVE_THRESHOLD,
            precision=0.0,
            recall=0.0,
            samples=len(labeled_tickets),
            accuracy=accuracy,
        )

    optimal_threshold, optimal_precision, optimal_recall = max(
        candidates, key=lambda item: item[2]
    )

    logger.info(
        "Оптимальный порог %.2f: precision=%.2f recall=%.2f на %d примерах",
        optimal_threshold,
        optimal_precision,
        optimal_recall,
        len(labeled_tickets),
    )

    return CalibrationReport(
        threshold=float(optimal_threshold),
        precision=float(optimal_precision),
        recall=float(optimal_recall),
        samples=len(labeled_tickets),
        accuracy=accuracy,
    )
