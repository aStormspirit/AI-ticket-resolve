from .classifier import classify_intent
from .policy_engine import policy_engine
from .ticket_resolver import auto_resolve


def process_ticket(ticket_text: str, ticket_id: str) -> dict:
    # Шаг 1: Классификация намерения
    intent = classify_intent(ticket_text)
    
    # Шаг 2: Принятие решения через policy engine
    decision = policy_engine.decide(intent, ticket_text)
    
    # Шаг 3: Немедленная эскалация если требуется
    if decision.should_escalate:
        return {
            "ticket_id": ticket_id,
            "status": "escalated",
            "reason": decision.escalation_reason.value,
            "intent": intent.intent,
            "priority": decision.priority.value,
            "assigned_to": decision.assigned_queue,
            "reasoning": decision.reasoning
        }
    
    # Шаг 4: Попытка автоматического разрешения
    if decision.can_use_tools:
        resolution = auto_resolve(ticket_text, intent.intent)
        
        # Шаг 5: Эскалация при неудаче инструментов
        if resolution.get("requires_escalation"):
            return {
                "ticket_id": ticket_id,
                "status": "escalated_tool_failure",
                "reason": "tool_failure",
                "intent": intent.intent,
                "attempted_resolution": resolution,
                "assigned_to": "manual_review_queue"
            }
        
        if resolution.get("resolved"):
            return {
                "ticket_id": ticket_id,
                "status": "auto_resolved",
                "resolution": resolution,
                "intent": intent.intent,
                "priority": decision.priority.value
            }
    
    # Fallback: требуется ручная обработка
    return {
        "ticket_id": ticket_id,
        "status": "requires_manual_handling",
        "intent": intent.intent,
        "assigned_to": decision.assigned_queue,
        "reasoning": decision.reasoning
    }
