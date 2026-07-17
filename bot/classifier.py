from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .llm import llm


class IntentClassification(BaseModel):
    intent: str = Field(description="Категория намерения пользователя")
    confidence: float = Field(description="Уверенность в классификации от 0 до 1")
    reasoning: str = Field(description="Объяснение почему выбрана эта категория")


# Определяем возможные намерения
INTENTS = {
    "password_reset": "Сброс или восстановление пароля",
    "order_status": "Проверка статуса заказа",
    "address_change": "Изменение адреса доставки",
    "refund_request": "Запрос возврата средств",
    "complaint": "Жалоба на качество товара или обслуживания",
    "general_inquiry": "Общий вопрос или другое",
}


parser = PydanticOutputParser(pydantic_object=IntentClassification)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Ты классификатор намерений для службы поддержки.
    
Доступные категории:
{intents}

Проанализируй запрос пользователя и определи его намерение.
Если уверенность ниже 0.7, выбирай general_inquiry.

{format_instructions}""",
        ),
        ("user", "{ticket_text}"),
    ]
)

chain = prompt | llm | parser


def classify_intent(ticket_text: str) -> IntentClassification:
    result = chain.invoke(
        {
            "ticket_text": ticket_text,
            "intents": "\n".join([f"- {k}: {v}" for k, v in INTENTS.items()]),
            "format_instructions": parser.get_format_instructions(),
        }
    )
    return result
