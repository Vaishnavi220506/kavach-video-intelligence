"""Deterministic routing, retrieval, and grounding tests for Module 10."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from kavach.assistant import (
    EVENT_EXPLANATION,
    EVENTS_AROUND_TIME,
    EVENTS_BY_ENTITY,
    EVENTS_BY_RISK,
    EVENTS_BY_TYPE,
    REVIEW_TIMESTAMPS,
    STATISTICS,
    GroundedAssistant,
    OllamaClient,
    OllamaResponse,
    QueryRouter,
    RetrievalService,
)
from kavach.behaviours import BehaviourEvent
from kavach.incidents import IncidentManager
from kavach.risk import RiskEngine
from kavach.storage import EventDatabase


class FakeOllama:
    """Small model double that records the supplied grounding context."""

    model = "fake:3b"

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []
        self.content = "The retrieved records support a cautious explanation."

    def chat(self, messages):
        normalized = [dict(message) for message in messages]
        self.messages.append(normalized)
        return OllamaResponse(
            model=self.model,
            content=self.content,
        )


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class AssistantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = EventDatabase(Path(self.temp_dir.name) / "events.sqlite3")
        self.database.register_video("video-1", "warehouse.mp4")
        self.manager = IncidentManager(risk_engine=RiskEngine())
        self._store(
            BehaviourEvent(
                "POSSIBLE_DRAGGING",
                12.0,
                ("carton_12", "person_3"),
                0.80,
                {
                    "near_floor": True,
                    "horizontal_displacement_px": 120.0,
                    "duration_seconds": 2.0,
                },
            )
        )
        self._store(
            BehaviourEvent(
                "POSSIBLE_DRAGGING",
                20.0,
                ("carton_7", "person_3"),
                0.80,
                {
                    "near_floor": True,
                    "horizontal_displacement_px": 120.0,
                    "duration_seconds": 2.0,
                },
            )
        )
        self._store(
            BehaviourEvent(
                "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
                720.0,
                ("person_3", "forklift_1"),
                0.91,
                {
                    "distance_px": 20.0,
                    "maximum_distance_px": 100.0,
                    "approaching": True,
                },
            )
        )
        self.fake_ollama = FakeOllama()
        self.assistant = GroundedAssistant(
            self.database,
            ollama_client=self.fake_ollama,
            retrieval=RetrievalService(
                self.database,
                default_video_id="video-1",
            ),
        )

    def tearDown(self) -> None:
        self.database.close()
        self.temp_dir.cleanup()

    def _store(self, event: BehaviourEvent) -> str:
        incident = self.manager.ingest(event).incident
        return self.database.insert_event("video-1", incident)

    def test_router_maps_supported_questions_before_llm(self) -> None:
        router = QueryRouter()

        self.assertEqual(
            router.route("Show all dragging incidents").kind,
            EVENTS_BY_TYPE,
        )
        self.assertEqual(
            router.route("Find every time a worker was close to a forklift").behaviour,
            "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
        )
        self.assertEqual(
            router.route("Show every motion anomaly").behaviour,
            "MOTION_ANOMALY",
        )
        self.assertEqual(
            router.route("Show object activity signals").behaviour,
            "OBJECT_ACTIVITY",
        )
        self.assertEqual(
            router.route("What happened around 12 minutes?").kind,
            EVENTS_AROUND_TIME,
        )
        self.assertEqual(
            router.route("Why was Event #32 considered high risk?").kind,
            EVENT_EXPLANATION,
        )
        self.assertEqual(
            router.route("Show incidents involving Carton #12").kind,
            EVENTS_BY_ENTITY,
        )
        self.assertEqual(
            router.route("Show all high risk events").kind,
            EVENTS_BY_RISK,
        )
        self.assertEqual(
            router.route("Give me the timestamps worth reviewing").kind,
            REVIEW_TIMESTAMPS,
        )
        self.assertEqual(
            router.route("Which behaviour occurred most frequently?").kind,
            STATISTICS,
        )

    def test_deterministic_type_query_returns_actual_references(self) -> None:
        response = self.assistant.ask("Show all dragging events.")

        self.assertFalse(response.used_llm)
        self.assertEqual(response.intent.kind, EVENTS_BY_TYPE)
        self.assertEqual(len(response.event_references), 2)
        self.assertIn("00:12", response.answer)
        self.assertIn("INC-000001", response.answer)
        self.assertIn("INC-000002", response.answer)

    def test_statistics_are_computed_without_llm_counting(self) -> None:
        response = self.assistant.ask("Which behaviour occurred most frequently?")

        self.assertFalse(response.used_llm)
        self.assertEqual(response.intent.kind, STATISTICS)
        self.assertIn("Most frequent behaviour: POSSIBLE_DRAGGING (2)", response.answer)
        self.assertEqual(self.fake_ollama.messages, [])

    def test_entity_and_review_queries_use_database_filters(self) -> None:
        entity_response = self.assistant.ask("Show incidents involving Carton #12.")
        review_response = self.assistant.ask("Give me the timestamps worth reviewing.")

        self.assertEqual(len(entity_response.events), 1)
        self.assertEqual(entity_response.events[0]["entities"][0], "carton_12")
        self.assertEqual(len(review_response.events), 1)
        self.assertEqual(
            review_response.events[0]["behaviour"],
            "UNSAFE_HUMAN_FORKLIFT_PROXIMITY",
        )

    def test_llm_receives_only_retrieved_context_and_references_are_appended(self) -> None:
        response = self.assistant.ask("What happened around 12 minutes?")

        self.assertTrue(response.used_llm)
        self.assertTrue(response.grounded)
        self.assertEqual(response.intent.kind, EVENTS_AROUND_TIME)
        self.assertIn("The retrieved records support", response.answer)
        self.assertIn("Verified event references:", response.answer)
        context_message = self.fake_ollama.messages[-1][1]["content"]
        self.assertIn("INC-000003", context_message)
        self.assertNotIn("raw video", context_message.lower())

    def test_event_explanation_is_grounded_to_stored_risk(self) -> None:
        response = self.assistant.ask("Why was Event #3 considered high risk?")

        self.assertTrue(response.used_llm)
        self.assertEqual(response.events[0]["risk"]["category"], "HIGH")
        self.assertIn("INC-000003", response.answer)

    def test_prohibited_model_language_falls_back_to_deterministic_facts(self) -> None:
        self.fake_ollama.content = (
            "The confidence indicates a strong likelihood of the event occurring."
        )

        response = self.assistant.ask("Why was Event #3 considered high risk?")

        self.assertFalse(response.used_llm)
        self.assertIn("Event INC-000003", response.answer)
        self.assertIn("prohibited", response.llm_error or "")

    def test_unknown_question_returns_insufficient_evidence(self) -> None:
        response = self.assistant.ask("Tell me an unrecorded story about the shift.")

        self.assertFalse(response.used_llm)
        self.assertIn("insufficient evidence", response.answer.lower())


class OllamaClientTests(unittest.TestCase):
    def test_http_client_parses_chat_tags_and_resource_usage(self) -> None:
        def opener(request, timeout):
            if request.full_url.endswith("/api/tags"):
                return FakeHTTPResponse({"models": [{"name": "llama3.2:3b"}]})
            if request.full_url.endswith("/api/ps"):
                return FakeHTTPResponse(
                    {
                        "models": [
                            {
                                "name": "llama3.2:3b",
                                "size": 2_000_000_000,
                                "size_vram": 0,
                            }
                        ]
                    }
                )
            return FakeHTTPResponse(
                {
                    "model": "llama3.2:3b",
                    "message": {"role": "assistant", "content": "grounded"},
                    "total_duration": 1_000_000,
                    "prompt_eval_count": 10,
                    "eval_count": 4,
                }
            )

        client = OllamaClient(opener=opener)
        messages = [{"role": "user", "content": "Use only this context."}]

        self.assertEqual(client.list_models(), ("llama3.2:3b",))
        response = client.chat(messages)
        resources = client.resource_usage()

        self.assertEqual(response.content, "grounded")
        self.assertEqual(response.eval_count, 4)
        self.assertEqual(resources["total_model_size_bytes"], 2_000_000_000)
        self.assertEqual(resources["total_vram_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
