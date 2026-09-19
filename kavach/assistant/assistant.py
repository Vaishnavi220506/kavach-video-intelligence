"""Grounded assistant orchestration: route, retrieve, then optionally explain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic

from ..incidents.replay import EvidenceReplay, ReplayResult
from ..storage import EventDatabase
from .ollama_client import (
    OllamaBenchmark,
    OllamaClient,
    OllamaError,
)
from .prompts import build_messages
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
    QueryIntent,
    QueryRouter,
)
from .retrieval import RetrievalError, RetrievalResult, RetrievalService, format_timestamp


class AssistantError(RuntimeError):
    """Raised when assistant orchestration or replay cannot proceed."""


@dataclass(frozen=True)
class AssistantResponse:
    """Grounded answer plus machine-readable event references."""

    question: str
    answer: str
    intent: QueryIntent
    events: tuple[dict[str, object], ...]
    event_references: tuple[dict[str, object], ...]
    statistics: Mapping[str, object] | None
    used_llm: bool
    grounded: bool
    latency_seconds: float
    llm_model: str | None = None
    llm_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return an API/UI-friendly response payload."""

        return {
            "question": self.question,
            "answer": self.answer,
            "intent": {
                "kind": self.intent.kind,
                "video_id": self.intent.video_id,
                "event_id": self.intent.event_id,
                "behaviour": self.intent.behaviour,
                "entity_id": self.intent.entity_id,
                "risk_category": self.intent.risk_category,
                "timestamp_seconds": self.intent.timestamp_seconds,
                "window_seconds": self.intent.window_seconds,
            },
            "events": list(self.events),
            "event_references": list(self.event_references),
            "statistics": (
                None if self.statistics is None else dict(self.statistics)
            ),
            "used_llm": self.used_llm,
            "grounded": self.grounded,
            "latency_seconds": self.latency_seconds,
            "llm_model": self.llm_model,
            "llm_error": self.llm_error,
        }


class GroundedAssistant:
    """Answer natural-language questions using SQLite facts and local Ollama."""

    _unsafe_llm_terms = (
        "likelihood",
        "probability",
        "damage",
        "damaged",
        "injury",
        "injured",
        "impact",
        "collision",
    )
    _deterministic_kinds = {
        EVENT_BY_ID,
        EVENTS_BY_ENTITY,
        EVENTS_BY_RISK,
        EVENTS_BY_TYPE,
        REVIEW_TIMESTAMPS,
        STATISTICS,
    }
    # Video-level summaries are exact aggregations, not a safe place for a
    # small local model to recount a long event list. Ollama remains useful
    # for bounded event explanations and time-window interpretation.
    _llm_kinds = {EVENT_EXPLANATION, EVENTS_AROUND_TIME}

    def __init__(
        self,
        database: EventDatabase,
        *,
        ollama_client: OllamaClient | None = None,
        retrieval: RetrievalService | None = None,
        router: QueryRouter | None = None,
        replay: EvidenceReplay | None = None,
    ) -> None:
        if not isinstance(database, EventDatabase):
            raise TypeError("database must be an EventDatabase")
        self.database = database
        self.client = ollama_client or OllamaClient()
        self.retrieval = retrieval or RetrievalService(database)
        self.router = router or QueryRouter()
        self.replay = replay

    @staticmethod
    def _event_label(event: Mapping[str, object]) -> str:
        return str(event.get("behaviour", "event"))

    @staticmethod
    def _reference_lines(
        references: tuple[dict[str, object], ...],
    ) -> list[str]:
        lines = ["Verified event references:"]
        for reference in references:
            clip = reference.get("clip_path")
            line = (
                f"- {reference.get('timestamp_display')} — "
                f"{reference.get('event_id')} — {reference.get('behaviour')}"
            )
            if clip:
                # Keep local filesystem paths out of a supervisor-facing
                # answer. The web client receives a separate replay action
                # tied to this event reference.
                line += " — replay available"
            lines.append(line)
        return lines

    @classmethod
    def _append_references(
        cls,
        answer: str,
        references: tuple[dict[str, object], ...],
    ) -> str:
        if not references:
            return answer.strip()
        return "\n".join(
            [answer.strip(), "", *cls._reference_lines(references)]
        ).strip()

    @classmethod
    def _accept_llm_text(cls, text: str) -> tuple[bool, str]:
        """Reject common unsupported claims that violate the grounding policy."""

        normalized = str(text).lower()
        if any(term in normalized for term in cls._unsafe_llm_terms):
            return (
                False,
                "LLM response contained a prohibited unsupported-outcome or "
                "confidence claim",
            )
        return True, ""

    @staticmethod
    def _list_answer(
        result: RetrievalResult,
        noun: str,
    ) -> str:
        if not result.events:
            return f"No stored {noun} were found in the supplied KAVACH records."
        return (
            f"Found {result.total_event_count} stored {noun}. "
            "The verified references are listed below."
        )

    @classmethod
    def _deterministic_answer(cls, result: RetrievalResult) -> str:
        intent = result.intent
        if intent.kind == EVENTS_BY_TYPE:
            return cls._list_answer(
                result,
                intent.behaviour or "matching behaviour",
            )
        if intent.kind == EVENTS_BY_ENTITY:
            return cls._list_answer(
                result,
                f"events involving {intent.entity_id}",
            )
        if intent.kind == EVENTS_BY_RISK:
            return cls._list_answer(
                result,
                f"{intent.risk_category} risk events",
            )
        if intent.kind == REVIEW_TIMESTAMPS:
            return cls._list_answer(result, "events worth reviewing")
        if intent.kind == EVENTS_AROUND_TIME:
            if not result.events:
                return (
                    "There is insufficient evidence in the stored KAVACH "
                    "records for that time window."
                )
            start = max(
                0.0,
                float(intent.timestamp_seconds or 0.0) - intent.window_seconds,
            )
            end = float(intent.timestamp_seconds or 0.0) + intent.window_seconds
            return (
                f"Found {result.total_event_count} stored events between "
                f"{format_timestamp(start)} and {format_timestamp(end)}."
            )
        if intent.kind in {EVENT_BY_ID, EVENT_EXPLANATION}:
            if not result.events:
                return (
                    "There is insufficient evidence in the stored KAVACH "
                    "records for that event ID."
                )
            event = result.events[0]
            risk = event.get("risk")
            if not isinstance(risk, Mapping):
                return (
                    f"Stored event {event.get('event_id')} has no structured "
                    "risk assessment."
                )
            breakdown = risk.get("breakdown")
            components = {}
            if isinstance(breakdown, Mapping):
                raw_components = breakdown.get("components")
                if isinstance(raw_components, Mapping):
                    components = raw_components
            component_text = ", ".join(
                f"{name}={value}"
                for name, value in components.items()
            )
            return (
                f"Event {event.get('event_id')} at "
                f"{format_timestamp(float(event.get('timestamp', 0.0)))} "
                f"was stored as {event.get('behaviour')} with risk "
                f"{risk.get('category')} ({risk.get('score')}/100). "
                f"Detection confidence was {risk.get('detection_confidence')}. "
                f"Weighted evidence components: {component_text or 'not supplied'}."
            )
        if intent.kind == STATISTICS:
            statistics = result.statistics or {}
            total = int(statistics.get("total_events", 0))
            by_behaviour = statistics.get("by_behaviour", {})
            if not isinstance(by_behaviour, Mapping) or not by_behaviour:
                return (
                    "There is insufficient evidence in the stored KAVACH "
                    "records to calculate behaviour frequency."
                )
            most_common = max(
                by_behaviour.items(),
                key=lambda item: int(item[1]),
            )
            counts = ", ".join(
                f"{name}: {count}" for name, count in by_behaviour.items()
            )
            return (
                f"The database contains {total} stored events. Most frequent "
                f"behaviour: {most_common[0]} ({most_common[1]}). "
                f"Counts: {counts}."
            )
        if intent.kind == SUMMARY:
            statistics = result.statistics or {}
            total = int(statistics.get("total_events", 0))
            if total == 0:
                return (
                    "There is insufficient evidence in the stored KAVACH "
                    "records to summarize this video."
                )
            by_behaviour = statistics.get("by_behaviour", {})
            by_risk = statistics.get("by_risk", {})
            behaviour_text = ", ".join(
                f"{str(name).replace('_', ' ').title()}: {count}"
                for name, count in by_behaviour.items()
            )
            risk_text = ", ".join(
                f"{str(name).title()}: {count}"
                for name, count in by_risk.items()
            )
            return (
                f"Video summary from stored KAVACH evidence: {total} event(s). "
                f"Behaviours — {behaviour_text or 'none recorded'}. "
                f"Risk distribution — {risk_text or 'none recorded'}."
            )
        return "There is insufficient evidence in the stored KAVACH records."

    def ask(
        self,
        question: str,
        *,
        video_id: str | None = None,
        use_llm: bool = True,
    ) -> AssistantResponse:
        """Route and retrieve first, then optionally ask Ollama to verbalize."""

        started = monotonic()
        intent = self.router.route(question, video_id=video_id)
        try:
            result = self.retrieval.retrieve(intent)
        except RetrievalError as exc:
            return AssistantResponse(
                question=str(question).strip(),
                answer=f"Insufficient retrieval context: {exc}",
                intent=intent,
                events=(),
                event_references=(),
                statistics=None,
                used_llm=False,
                grounded=True,
                latency_seconds=monotonic() - started,
            )

        references = result.event_references()
        fallback = self._deterministic_answer(result)
        should_call_llm = (
            use_llm
            and intent.kind in self._llm_kinds
            and bool(result.events or result.statistics)
        )
        answer = fallback
        used_llm = False
        llm_error = None
        llm_model = None
        if should_call_llm:
            try:
                response = self.client.chat(
                    build_messages(question, result.context_dict())
                )
                accepted, rejection_reason = self._accept_llm_text(response.content)
                if accepted:
                    answer = response.content
                    used_llm = True
                    llm_model = response.model
                else:
                    llm_error = rejection_reason
            except OllamaError as exc:
                llm_error = str(exc)
        answer = self._append_references(answer, references)
        return AssistantResponse(
            question=str(question).strip(),
            answer=answer,
            intent=intent,
            events=result.events,
            event_references=references,
            statistics=result.statistics,
            used_llm=used_llm,
            grounded=True,
            latency_seconds=monotonic() - started,
            llm_model=llm_model,
            llm_error=llm_error,
        )

    def benchmark(
        self,
        question: str,
        *,
        video_id: str | None = None,
        repeats: int = 1,
    ) -> OllamaBenchmark:
        """Benchmark Ollama on the retrieved context for one question."""

        intent = self.router.route(question, video_id=video_id)
        result = self.retrieval.retrieve(intent)
        return self.client.benchmark(
            build_messages(question, result.context_dict()),
            repeats=repeats,
        )

    def replay_event(
        self,
        event_id: str,
        *,
        force: bool = False,
    ) -> ReplayResult:
        """Create or reuse the Module 9 evidence clip for an event reference."""

        if self.replay is None:
            raise AssistantError(
                "no EvidenceReplay instance was supplied to this assistant"
            )
        return self.replay.create_clip(event_id, force=force)
