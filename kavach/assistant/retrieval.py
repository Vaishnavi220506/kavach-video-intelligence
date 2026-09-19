"""Deterministic retrieval of grounded event records from SQLite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json

from ..storage import EventDatabase
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
)


DEFAULT_OPERATIONAL_RULES = {
    "evidence_clip_window_seconds": {"before": 3.0, "after": 3.0},
    "timestamps_worth_reviewing_minimum_risk_score": 50.0,
    "risk_categories": ("LOW", "MEDIUM", "HIGH", "CRITICAL"),
    "grounding_note": (
        "Factual claims must be limited to the supplied stored event records, "
        "risk evidence, statistics, and these configured rules."
    ),
}


class RetrievalError(RuntimeError):
    """Raised when a grounded retrieval request cannot be fulfilled."""


def format_timestamp(seconds: float) -> str:
    """Format source seconds as MM:SS or HH:MM:SS."""

    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


@dataclass(frozen=True)
class RetrievalResult:
    """Rows and statistics retrieved for one deterministic query plan."""

    intent: QueryIntent
    events: tuple[dict[str, object], ...]
    total_event_count: int
    statistics: Mapping[str, object] | None = None
    truncated: bool = False
    operational_rules: Mapping[str, object] = field(
        default_factory=lambda: dict(DEFAULT_OPERATIONAL_RULES)
    )

    def event_references(self) -> tuple[dict[str, object], ...]:
        """Return compact, replay-oriented references for user responses."""

        references = []
        for event in self.events:
            risk = event.get("risk")
            risk_score = None
            risk_category = None
            if isinstance(risk, Mapping):
                risk_score = risk.get("score")
                risk_category = risk.get("category")
            references.append(
                {
                    "event_id": event.get("event_id"),
                    "timestamp": event.get("timestamp"),
                    "timestamp_display": format_timestamp(
                        float(event.get("timestamp", 0.0))
                    ),
                    "behaviour": event.get("behaviour"),
                    "entities": event.get("entities", []),
                    "risk_score": risk_score,
                    "risk_category": risk_category,
                    "review_status": event.get("review_status"),
                    "clip_path": event.get("clip_path"),
                }
            )
        return tuple(references)

    def context_dict(self) -> dict[str, object]:
        """Return only retrieved data and configured rules for an LLM prompt."""

        events: list[dict[str, object]] = list(self.events)
        if self.intent.kind == SUMMARY:
            # A compact summary context is easier for a small local model to
            # follow than every nested evidence field. Detailed event
            # evidence is still retrieved for event-specific questions.
            events = [
                {
                    "event_id": event.get("event_id"),
                    "timestamp": event.get("timestamp"),
                    "behaviour": event.get("behaviour"),
                    "entities": event.get("entities", []),
                    "risk_score": (
                        event.get("risk", {}).get("score")
                        if isinstance(event.get("risk"), Mapping)
                        else None
                    ),
                    "risk_category": (
                        event.get("risk", {}).get("category")
                        if isinstance(event.get("risk"), Mapping)
                        else None
                    ),
                    "review_status": event.get("review_status"),
                }
                for event in self.events
            ]

        return {
            "query_intent": {
                "kind": self.intent.kind,
                "video_id": self.intent.video_id,
                "event_id": self.intent.event_id,
                "behaviour": self.intent.behaviour,
                "entity_id": self.intent.entity_id,
                "risk_category": self.intent.risk_category,
                "timestamp_seconds": self.intent.timestamp_seconds,
                "window_seconds": self.intent.window_seconds,
            },
            "events": events,
            "total_event_count": self.total_event_count,
            "statistics": (
                None if self.statistics is None else dict(self.statistics)
            ),
            "truncated": self.truncated,
            "configured_operational_rules": dict(self.operational_rules),
        }

    def context_json(self) -> str:
        """Serialize the grounding context for a local model prompt."""

        return json.dumps(self.context_dict(), ensure_ascii=False, sort_keys=True)


class RetrievalService:
    """Execute QueryIntent objects against the Module 9 EventDatabase."""

    def __init__(
        self,
        database: EventDatabase,
        *,
        default_video_id: str | None = None,
        max_context_events: int = 100,
        operational_rules: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(database, EventDatabase):
            raise TypeError("database must be an EventDatabase")
        if int(max_context_events) <= 0:
            raise RetrievalError("max_context_events must be positive")
        self.database = database
        self.default_video_id = default_video_id
        self.max_context_events = int(max_context_events)
        self.operational_rules = dict(
            DEFAULT_OPERATIONAL_RULES
            if operational_rules is None
            else operational_rules
        )

    def _resolve_video_id(self, video_id: str | None) -> str:
        if video_id:
            return str(video_id)
        if self.default_video_id:
            return str(self.default_video_id)
        videos = self.database.list_videos()
        if len(videos) == 1:
            return str(videos[0]["video_id"])
        if not videos:
            raise RetrievalError("no registered video is available")
        raise RetrievalError(
            "video_id is required because multiple registered videos are available"
        )

    def _limit(
        self,
        events: list[dict[str, object]],
    ) -> tuple[tuple[dict[str, object], ...], bool]:
        return (
            tuple(events[: self.max_context_events]),
            len(events) > self.max_context_events,
        )

    def retrieve(self, intent: QueryIntent) -> RetrievalResult:
        """Run the routed query without asking Ollama to search or count."""

        if not isinstance(intent, QueryIntent):
            raise TypeError("intent must be a QueryIntent")
        video_id = intent.video_id
        events: list[dict[str, object]]
        statistics: Mapping[str, object] | None = None

        if intent.kind in {EVENT_BY_ID, EVENT_EXPLANATION}:
            if not intent.event_id:
                raise RetrievalError("event query has no event_id")
            event = self.database.get_event(intent.event_id)
            events = [] if event is None else [event]
        elif intent.kind == EVENTS_AROUND_TIME:
            if intent.timestamp_seconds is None:
                raise RetrievalError("time query has no timestamp")
            resolved_video = self._resolve_video_id(video_id)
            start = max(0.0, intent.timestamp_seconds - intent.window_seconds)
            end = intent.timestamp_seconds + intent.window_seconds
            events = self.database.get_events_between_times(
                resolved_video,
                start,
                end,
            )
        elif intent.kind == EVENTS_BY_TYPE:
            events = self.database.get_events_by_type(
                intent.behaviour or "",
                video_id=video_id,
            )
        elif intent.kind == EVENTS_BY_ENTITY:
            events = self.database.get_events_for_entity(
                intent.entity_id or "",
                video_id=video_id,
            )
        elif intent.kind == EVENTS_BY_RISK:
            events = self.database.get_events_by_risk(
                intent.risk_category,
                video_id=video_id,
            )
        elif intent.kind == REVIEW_TIMESTAMPS:
            events = self.database.get_events_by_risk(
                minimum_score=float(
                    self.operational_rules.get(
                        "timestamps_worth_reviewing_minimum_risk_score",
                        50.0,
                    )
                ),
                video_id=video_id,
            )
        elif intent.kind == STATISTICS:
            statistics = self.database.get_event_statistics(video_id=video_id)
            events = []
        elif intent.kind == SUMMARY:
            resolved_video = self._resolve_video_id(video_id)
            all_events = self.database.get_events(resolved_video)
            statistics = self.database.get_event_statistics(video_id=resolved_video)
            events = all_events
        else:
            events = []

        limited, truncated = self._limit(events)
        return RetrievalResult(
            intent=intent,
            events=limited,
            total_event_count=len(events) if statistics is None else int(
                statistics.get("total_events", len(events))
            ),
            statistics=statistics,
            truncated=truncated,
            operational_rules=self.operational_rules,
        )
