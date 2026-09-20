"""Deterministic routing from supported natural-language questions to queries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

EVENT_BY_ID = "EVENT_BY_ID"
EVENT_EXPLANATION = "EVENT_EXPLANATION"
EVENTS_AROUND_TIME = "EVENTS_AROUND_TIME"
EVENTS_BY_ENTITY = "EVENTS_BY_ENTITY"
EVENTS_BY_RISK = "EVENTS_BY_RISK"
EVENTS_BY_TYPE = "EVENTS_BY_TYPE"
REVIEW_TIMESTAMPS = "REVIEW_TIMESTAMPS"
STATISTICS = "STATISTICS"
SUMMARY = "SUMMARY"
UNKNOWN = "UNKNOWN"

QUERY_KINDS = (
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
)

QueryKind = Literal[
    "EVENT_BY_ID",
    "EVENT_EXPLANATION",
    "EVENTS_AROUND_TIME",
    "EVENTS_BY_ENTITY",
    "EVENTS_BY_RISK",
    "EVENTS_BY_TYPE",
    "REVIEW_TIMESTAMPS",
    "STATISTICS",
    "SUMMARY",
    "UNKNOWN",
]


class QueryRouterError(ValueError):
    """Raised when a natural-language query cannot be routed safely."""


@dataclass(frozen=True)
class QueryIntent:
    """A deterministic query plan produced before any LLM call."""

    kind: QueryKind
    raw_query: str
    video_id: str | None = None
    event_id: str | None = None
    behaviour: str | None = None
    entity_id: str | None = None
    risk_category: str | None = None
    timestamp_seconds: float | None = None
    window_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.kind not in QUERY_KINDS:
            raise QueryRouterError(f"unknown query kind: {self.kind}")
        if not str(self.raw_query).strip():
            raise QueryRouterError("query cannot be empty")
        if not math.isfinite(float(self.window_seconds)) or self.window_seconds < 0:
            raise QueryRouterError("window_seconds must be finite and non-negative")
        if self.timestamp_seconds is not None:
            timestamp = float(self.timestamp_seconds)
            if not math.isfinite(timestamp) or timestamp < 0.0:
                raise QueryRouterError(
                    "timestamp_seconds must be finite and non-negative"
                )
            object.__setattr__(self, "timestamp_seconds", timestamp)
        if self.video_id is not None:
            object.__setattr__(self, "video_id", str(self.video_id).strip())
        if self.event_id is not None:
            object.__setattr__(self, "event_id", str(self.event_id).strip())
        if self.behaviour is not None:
            object.__setattr__(self, "behaviour", str(self.behaviour).strip().upper())
        if self.entity_id is not None:
            object.__setattr__(
                self,
                "entity_id",
                _normalize_entity_id(self.entity_id),
            )
        if self.risk_category is not None:
            category = str(self.risk_category).strip().upper()
            if category not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                raise QueryRouterError(f"unknown risk category: {category}")
            object.__setattr__(self, "risk_category", category)
        object.__setattr__(self, "window_seconds", float(self.window_seconds))


def _normalize_entity_id(value: object) -> str:
    normalized = "_".join(
        str(value).strip().lower().replace("-", " ").replace("#", " ").split()
    )
    return normalized


class QueryRouter:
    """Recognize the narrow, auditable question vocabulary for Module 10."""

    _event_id_patterns = (
        re.compile(r"\binc[-_ ]?0*(\d+)\b", re.IGNORECASE),
        re.compile(r"\bevent\s*#?\s*0*(\d+)\b", re.IGNORECASE),
    )
    _time_pattern = re.compile(
        r"\b(?:around|near|at|about)\s+"
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
        re.IGNORECASE,
    )
    _entity_pattern = re.compile(
        r"\b(pallet[ _-]+truck|pallet|carton|box|package|person|worker|"
        r"forklift|trolley|truck)[\s_#-]*(\d+)\b",
        re.IGNORECASE,
    )

    def __init__(self, *, around_window_seconds: float = 10.0) -> None:
        window = float(around_window_seconds)
        if not math.isfinite(window) or window < 0.0:
            raise QueryRouterError(
                "around_window_seconds must be finite and non-negative"
            )
        self.around_window_seconds = window

    @staticmethod
    def _parse_time(value: float, unit: str) -> float:
        normalized = unit.lower()
        if normalized.startswith("hour") or normalized in {"h", "hrs"}:
            multiplier = 3600.0
        elif normalized.startswith("min") or normalized == "m":
            multiplier = 60.0
        else:
            multiplier = 1.0
        return float(value) * multiplier

    @classmethod
    def _event_id(cls, query: str) -> str | None:
        for pattern in cls._event_id_patterns:
            match = pattern.search(query)
            if match:
                return f"INC-{int(match.group(1)):06d}"
        return None

    @classmethod
    def _timestamp(cls, query: str) -> float | None:
        match = cls._time_pattern.search(query)
        if not match:
            return None
        return cls._parse_time(float(match.group(1)), match.group(2))

    @classmethod
    def _entity_id(cls, query: str) -> str | None:
        match = cls._entity_pattern.search(query)
        if not match:
            return None
        return _normalize_entity_id(f"{match.group(1)}_{match.group(2)}")

    @staticmethod
    def _behaviour(query: str) -> str | None:
        normalized = query.lower()
        if (
            "close to a forklift" in normalized
            or "near a forklift" in normalized
            or "human forklift" in normalized
            or "worker forklift" in normalized
            or "forklift proximity" in normalized
            or "proximity" in normalized
        ):
            return "UNSAFE_HUMAN_FORKLIFT_PROXIMITY"
        aliases = (
            ("motion anomaly", "MOTION_ANOMALY"),
            ("anomal", "MOTION_ANOMALY"),
            ("unusual movement", "MOTION_ANOMALY"),
            ("sudden movement", "MOTION_ANOMALY"),
            ("unexpected movement", "MOTION_ANOMALY"),
            ("zone transition", "ZONE_TRANSITION"),
            ("entered zone", "ZONE_TRANSITION"),
            ("exited zone", "ZONE_TRANSITION"),
            ("activity signal", "OBJECT_ACTIVITY"),
            ("object activity", "OBJECT_ACTIVITY"),
            ("moving objects", "OBJECT_ACTIVITY"),
            ("drag", "POSSIBLE_DRAGGING"),
            ("drop", "POSSIBLE_DROP"),
            ("overhang", "PALLET_OVERHANG"),
            ("unstable stack", "UNSTABLE_STACK"),
            ("zone violation", "ZONE_VIOLATION"),
            ("restricted zone", "ZONE_VIOLATION"),
        )
        for phrase, behaviour in aliases:
            if phrase in normalized:
                return behaviour
        return None

    def route(
        self,
        query: str,
        *,
        video_id: str | None = None,
    ) -> QueryIntent:
        """Return a query plan without asking the LLM to interpret facts."""

        raw = str(query).strip()
        if not raw:
            raise QueryRouterError("query cannot be empty")
        normalized = " ".join(raw.lower().split())
        event_id = self._event_id(raw)
        if event_id is not None:
            kind = (
                EVENT_EXPLANATION
                if any(word in normalized for word in ("why", "risk", "explain"))
                else EVENT_BY_ID
            )
            return QueryIntent(
                kind=kind,
                raw_query=raw,
                video_id=video_id,
                event_id=event_id,
            )

        if any(
            phrase in normalized
            for phrase in (
                "summarize",
                "summary",
                "overview",
                "what happened in this video",
            )
        ):
            return QueryIntent(kind=SUMMARY, raw_query=raw, video_id=video_id)

        # The web assistant can answer broader video-level questions without
        # giving the model authority to search the database. Route only clear
        # video-overview language to the existing grounded SUMMARY retrieval;
        # unrelated questions remain UNKNOWN and therefore produce an
        # insufficient-evidence response.
        video_overview_language = any(
            phrase in normalized
            for phrase in (
                "explain this video",
                "explain the video",
                "explain everything",
                "explain all",
                "about this video",
                "about the video",
                "what happened in the video",
                "what happened overall",
                "overall safety",
                "overall picture",
                "main safety",
                "key findings",
                "entire video",
                "what should i know",
                "tell me about this video",
            )
        )
        if video_id and video_overview_language:
            return QueryIntent(kind=SUMMARY, raw_query=raw, video_id=video_id)

        if any(
            phrase in normalized
            for phrase in (
                "timestamps worth reviewing",
                "timestamps to review",
                "what should i review",
                "worth reviewing",
            )
        ):
            return QueryIntent(
                kind=REVIEW_TIMESTAMPS,
                raw_query=raw,
                video_id=video_id,
            )

        if "most frequently" in normalized or "most common" in normalized:
            return QueryIntent(kind=STATISTICS, raw_query=raw, video_id=video_id)

        behaviour = self._behaviour(raw)
        timestamp = self._timestamp(raw)
        if timestamp is not None:
            return QueryIntent(
                kind=EVENTS_AROUND_TIME,
                raw_query=raw,
                video_id=video_id,
                timestamp_seconds=timestamp,
                window_seconds=self.around_window_seconds,
            )

        entity_id = self._entity_id(raw)
        if entity_id is not None:
            return QueryIntent(
                kind=EVENTS_BY_ENTITY,
                raw_query=raw,
                video_id=video_id,
                entity_id=entity_id,
            )

        for category in ("critical", "high", "medium", "low"):
            if f"{category} risk" in normalized or normalized == category:
                return QueryIntent(
                    kind=EVENTS_BY_RISK,
                    raw_query=raw,
                    video_id=video_id,
                    risk_category=category.upper(),
                )

        if behaviour is not None:
            return QueryIntent(
                kind=EVENTS_BY_TYPE,
                raw_query=raw,
                video_id=video_id,
                behaviour=behaviour,
            )

        if any(
            phrase in normalized
            for phrase in (
                "how many",
                "count",
                "statistics",
                "statistic",
                "frequency",
                "frequencies",
            )
        ):
            return QueryIntent(kind=STATISTICS, raw_query=raw, video_id=video_id)

        return QueryIntent(kind=UNKNOWN, raw_query=raw, video_id=video_id)
