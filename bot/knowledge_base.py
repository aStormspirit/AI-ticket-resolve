"""База знаний с graceful degradation (spec §36).

Недоступность хранилища не останавливает обработку: агент продолжает работу без
поиска по документации, а факт деградации попадает в метрики.

Реализация по умолчанию — локальный индекс по ключевым словам. Класс
KnowledgeBase описывает контракт, к которому подключается векторное хранилище
(Qdrant/Chroma) без изменений в вызывающем коде.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from observability.metrics import degraded_operations_total

from .resilience import knowledge_base_breaker

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

_WORD = re.compile(r"\w+", re.UNICODE)


class KnowledgeBaseUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    keywords: tuple[str, ...]


class KnowledgeBase:
    """Локальный индекс по ключевым словам."""

    def __init__(self, source: Path = DEFAULT_SOURCE) -> None:
        self._source = source
        self._documents: list[Document] | None = None

    def _load(self) -> list[Document]:
        if self._documents is None:
            try:
                raw = json.loads(self._source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise KnowledgeBaseUnavailable(str(exc)) from exc
            self._documents = [
                Document(
                    id=item["id"],
                    title=item["title"],
                    content=item["content"],
                    keywords=tuple(item.get("keywords", [])),
                )
                for item in raw
            ]
        return self._documents

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        documents = self._load()
        tokens = {token.lower() for token in _WORD.findall(query)}

        scored: list[tuple[int, Document]] = []
        for document in documents:
            score = sum(
                1
                for keyword in document.keywords
                if keyword in tokens or any(keyword in token for token in tokens)
            )
            if score:
                scored.append((score, document))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [document for _, document in scored[:k]]


@lru_cache
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()


async def search(query: str, k: int = 3) -> list[Document]:
    """Поиск под защитой circuit breaker; при недоступности возвращает пустой список."""
    try:
        return await knowledge_base_breaker.call(
            lambda: _search_async(query, k),
        )
    except Exception:
        degraded_operations_total.labels(component="knowledge_base").inc()
        logger.warning(
            "База знаний недоступна, обработка продолжается без поиска по документации"
        )
        return []


async def _search_async(query: str, k: int) -> list[Document]:
    return get_knowledge_base().similarity_search(query, k)
