import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError(
        "Укажите OPENROUTER_API_KEY (или API_KEY) в .env"
    )

# GPT-5 через OpenRouter: https://openrouter.ai/openai/gpt-5
llm = ChatOpenRouter(
    model="openai/gpt-5",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)
