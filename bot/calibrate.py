from sklearn.metrics import precision_recall_curve

from .classifier import classify_intent


def calibrate_confidence_threshold(labeled_tickets: list[tuple[str, str]]) -> float:
    """
    Находит оптимальный порог уверенности на размеченных данных.
    
    Args:
        labeled_tickets: список пар (текст тикета, истинное намерение)
    
    Returns:
        Оптимальный порог уверенности для автоматической обработки
    """
    predictions = []
    ground_truth = []
    
    for ticket_text, true_intent in labeled_tickets:
        intent = classify_intent(ticket_text)
        predictions.append({
            'confidence': intent.confidence,
            'correct': intent.intent == true_intent
        })
        ground_truth.append(intent.intent == true_intent)
    
    confidences = [p['confidence'] for p in predictions]
    
    # Строим precision-recall кривую
    precision, recall, thresholds = precision_recall_curve(
        ground_truth,
        confidences
    )
    
    # Находим порог с балансом precision/recall
    # Например, минимум 90% precision для автообработки
    min_precision = 0.9
    valid_thresholds = thresholds[precision[:-1] >= min_precision]
    
    if len(valid_thresholds) == 0:
        return 0.95  # Консервативное значение
    
    # Выбираем порог с максимальным recall при заданной precision
    optimal_threshold = valid_thresholds[0]
    
    print(f"Оптимальный порог: {optimal_threshold:.2f}")
    print(f"При этом пороге precision={precision[0]:.2f}, recall={recall[0]:.2f}")
    
    return optimal_threshold

# Использование:
# optimal_threshold = calibrate_confidence_threshold(test_dataset)
# PolicyEngine.CONFIDENCE_THRESHOLD = optimal_threshold
