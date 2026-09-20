"""Grounded local Ollama assistant for KAVACH Module 10."""

from .assistant import AssistantError, AssistantResponse, GroundedAssistant
from .ollama_client import (
    OllamaBenchmark,
    OllamaClient,
    OllamaError,
    OllamaModelError,
    OllamaResponse,
    OllamaUnavailableError,
)
from .query_router import (
    EVENT_BY_ID,
    EVENT_EXPLANATION,
    EVENTS_AROUND_TIME,
    EVENTS_BY_ENTITY,
    EVENTS_BY_RISK,
    EVENTS_BY_TYPE,
    REVIEW_TIMESTAMPS,
    STATISTICS,
    SUMMARY,
    UNKNOWN,
    QueryIntent,
    QueryRouter,
    QueryRouterError,
)
from .retrieval import (
    RetrievalError,
    RetrievalResult,
    RetrievalService,
    format_timestamp,
)

__all__ = [
    "EVENTS_AROUND_TIME",
    "EVENTS_BY_ENTITY",
    "EVENTS_BY_RISK",
    "EVENTS_BY_TYPE",
    "EVENT_BY_ID",
    "EVENT_EXPLANATION",
    "REVIEW_TIMESTAMPS",
    "STATISTICS",
    "SUMMARY",
    "UNKNOWN",
    "AssistantError",
    "AssistantResponse",
    "GroundedAssistant",
    "OllamaBenchmark",
    "OllamaClient",
    "OllamaError",
    "OllamaModelError",
    "OllamaResponse",
    "OllamaUnavailableError",
    "QueryIntent",
    "QueryRouter",
    "QueryRouterError",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalService",
    "format_timestamp",
]
