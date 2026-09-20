"""Rule-based operational recommendations derived from incident evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class PreventionRecommendation:
    """One auditable recommendation; no generative model is involved."""

    rule_id: str
    root_cause: str
    title: str
    action: str
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "root_cause": self.root_cause,
            "title": self.title,
            "action": self.action,
            "rationale": self.rationale,
        }


class PreventionEngine:
    """Map structured behaviour evidence to cautious SOP-oriented actions."""

    _REVIEW_ONLY_TYPES = frozenset({"OBJECT_ACTIVITY", "ZONE_TRANSITION"})

    _RULES: dict[str, PreventionRecommendation] = {
        "POSSIBLE_THROWING": PreventionRecommendation(
            "P-THROW-01", "process", "Use a controlled hand-off", 
            "Require cartons to be placed or passed at walking speed; do not throw packages.",
            "The observed object showed a high image-space release speed or acceleration.",
        ),
        "POSSIBLE_ROUGH_HANDLING": PreventionRecommendation(
            "P-ROUGH-01", "process", "Reduce handling impulse",
            "Review the handling step and add a two-handed placement or soft-drop SOP where appropriate.",
            "The tracked object showed an abrupt acceleration or direction change.",
        ),
        "AISLE_OBSTRUCTION": PreventionRecommendation(
            "P-AISLE-01", "layout", "Keep travel paths clear",
            "Mark the aisle boundary and require unattended objects to be moved to a staging zone.",
            "An object remained stationary inside a configured aisle polygon.",
        ),
        "IMPROPER_PLACEMENT": PreventionRecommendation(
            "P-PLACE-01", "process", "Use designated staging locations",
            "Add a location check to the put-away step and configure the allowed zone for this camera.",
            "An object settled outside the configured allowed placement zones.",
        ),
        "COLLISION_RISK": PreventionRecommendation(
            "P-COLLISION-01", "layout", "Separate people and moving equipment",
            "Review the traffic plan, visibility at the crossing, and the configured warning distance.",
            "Tracked entities were close while their image-space distance was closing.",
        ),
        "UNSAFE_HUMAN_FORKLIFT_PROXIMITY": PreventionRecommendation(
            "P-PROX-01", "layout", "Reinforce pedestrian separation",
            "Review pedestrian walkways, forklift exclusion zones, and spotter requirements.",
            "A moving human/equipment pair entered the configured proximity threshold.",
        ),
        "POSSIBLE_DROP": PreventionRecommendation(
            "P-DROP-01", "process", "Use controlled lowering",
            "Review the lift/lower step and inspect the linked replay before assigning a physical cause.",
            "The temporal pattern was consistent with separation, downward motion, and settling.",
        ),
        "PALLET_OVERHANG": PreventionRecommendation(
            "P-SUPPORT-01", "process", "Center the load on its support",
            "Add a visual support-overlap check before the pallet is moved.",
            "The support ratio was below the configured geometric threshold.",
        ),
        "UNSTABLE_STACK": PreventionRecommendation(
            "P-STACK-01", "process", "Build a supported stack",
            "Review the stack base and require stable support overlap before adding another tier.",
            "At least one adjacent stack link had weak support geometry.",
        ),
    }

    def recommend(self, incident: Mapping[str, object]) -> list[dict[str, str]]:
        """Return recommendations for a stored/in-memory incident."""

        event_type = str(incident.get("type") or incident.get("behaviour") or "")
        # Activity transitions are useful context, but they are not safety
        # findings. Keeping them out of the prevention card makes the review
        # UI easier to read and avoids implying that ordinary movement needs a
        # corrective action.
        if event_type in self._REVIEW_ONLY_TYPES:
            return []
        recommendation = self._RULES.get(event_type)
        if recommendation is None:
            recommendation = PreventionRecommendation(
                "P-REVIEW-01",
                "unknown",
                "Review the linked evidence",
                "Keep the event in the supervisor review queue until the camera-specific rule is validated.",
                "No prevention rule is enabled for this event type.",
            )
        return [recommendation.to_dict()]

    def root_cause(self, incident: Mapping[str, object]) -> str:
        """Return the transparent category used by the matched rule."""

        recommendations = self.recommend(incident)
        return str(recommendations[0]["root_cause"]) if recommendations else "context"
