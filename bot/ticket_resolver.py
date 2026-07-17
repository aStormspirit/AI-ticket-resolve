import re

from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, EmailStr, field_validator

from .llm import llm


class PasswordResetInput(BaseModel):
    """Схема валидации для сброса пароля"""

    email: EmailStr


class OrderStatusInput(BaseModel):
    """Схема валидации для проверки заказа"""

    order_id: str

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, v: str) -> str:
        if not re.match(r"^ORD-\d{6}$", v):
            raise ValueError("Order ID must be in format ORD-XXXXXX")
        return v


@tool(args_schema=PasswordResetInput)
def send_password_reset(email: str) -> dict:
    """Отправляет ссылку для сброса пароля на указанный email."""
    try:
        send_reset_link_via_api(email)
        return {
            "success": True,
            "message": f"Ссылка для сброса пароля отправлена на {email}",
            "email": email,
        }
    except APIError as e:
        return {
            "success": False,
            "error": str(e),
            "should_escalate": True,
        }


@tool(args_schema=OrderStatusInput)
def check_order_status(order_id: str) -> dict:
    """Проверяет статус заказа по его номеру."""
    try:
        status_data = orders_api.get_status(order_id)
        return {
            "success": True,
            "order_id": order_id,
            "status": status_data["status"],
            "estimated_delivery": status_data.get("estimated_delivery"),
            "message": f"Заказ {order_id} находится в статусе: {status_data['status']}",
        }
    except OrderNotFoundError:
        return {
            "success": False,
            "error": f"Заказ {order_id} не найден",
            "should_escalate": True,
        }
    except APIError:
        return {
            "success": False,
            "error": "Система заказов временно недоступна",
            "should_escalate": True,
        }


@tool
def search_knowledge_base(query: str) -> dict:
    """Ищет информацию в базе знаний компании."""
    try:
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return {
                "success": False,
                "message": "Релевантная информация не найдена",
                "should_escalate": False,
            }
        return {
            "success": True,
            "content": "\n".join([doc.page_content for doc in results]),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "should_escalate": False,
        }


tools = [send_password_reset, check_order_status, search_knowledge_base]

SYSTEM_PROMPT = """Ты агент службы поддержки. Твоя задача — помочь клиенту решить проблему.

Используй доступные инструменты для решения запроса.
ВАЖНО: если инструмент вернул success=False и should_escalate=True, прекрати попытки и объясни, что нужна помощь оператора.
Будь вежливым и конкретным в ответах.

Текущий запрос был классифицирован как: {intent}"""


def auto_resolve(ticket_text: str, intent: str) -> dict:
    try:
        agent = create_agent(
            llm,
            tools,
            system_prompt=SYSTEM_PROMPT.format(intent=intent),
        )
        result = agent.invoke({"messages": [("user", ticket_text)]})
        messages = result.get("messages", [])

        tools_used = [
            msg.name
            for msg in messages
            if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None)
        ]
        has_tool_failures = any(
            isinstance(getattr(msg, "content", None), str)
            and '"should_escalate": true' in msg.content.lower()
            for msg in messages
            if getattr(msg, "type", None) == "tool"
        )

        last_ai = next(
            (msg for msg in reversed(messages) if getattr(msg, "type", None) == "ai"),
            None,
        )
        response = getattr(last_ai, "content", "") if last_ai else ""

        return {
            "resolved": not has_tool_failures,
            "response": response,
            "tools_used": tools_used,
            "requires_escalation": has_tool_failures,
        }
    except Exception as e:
        return {
            "resolved": False,
            "error": str(e),
            "requires_escalation": True,
        }
