"""Focused tests for Tutor Router V2 behavior.

This script avoids external API calls by patching `tutor_agent.llm`.
Run with:
    python test_tutor_router_v2.py
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

import tutor_agent
from student_store import default_student_model, ensure_model_shape


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])

    def invoke(self, _messages: Any) -> _FakeResponse:
        content = self._responses.pop(0) if self._responses else "{}"
        return _FakeResponse(content)


@contextmanager
def _patched_llm(responses: list[str] | None = None) -> Iterator[None]:
    original = tutor_agent.llm
    tutor_agent.llm = _FakeLLM(responses)
    try:
        yield
    finally:
        tutor_agent.llm = original


@contextmanager
def _patched_store_save() -> Iterator[None]:
    original = tutor_agent.store_save_student_model
    tutor_agent.store_save_student_model = lambda **_: None
    try:
        yield
    finally:
        tutor_agent.store_save_student_model = original


def _pending_model() -> dict[str, Any]:
    model = ensure_model_shape(default_student_model())
    model["active_review"] = {
        "status": "awaiting_answer",
        "awaiting_answer": True,
        "topic": "Self Attention",
        "concept_id": "attention_mechanism",
        "label": "Attention Mechanism",
        "question": "How does attention help with long sequences?",
        "mode": "prereq_gate",
        "goal_concept": "self_attention",
    }
    return model


def _route_state(student_text: str, model: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [],
        "current_topic": "Self Attention",
        "student_answer": student_text,
        "student_model": model,
    }


def test_route_turn_pending_social() -> None:
    with _patched_llm(["{}"]):
        out = tutor_agent.route_turn(_route_state("how are you", _pending_model()))
    route = out.get("route", {})
    assert route.get("intent") == "pending_social", route


def test_route_turn_pending_inability() -> None:
    with _patched_llm(["{}"]):
        out = tutor_agent.route_turn(_route_state("I can't", _pending_model()))
    route = out.get("route", {})
    assert route.get("intent") == "pending_answer", route


def test_route_turn_pending_switch() -> None:
    with _patched_llm(["{}"]):
        out = tutor_agent.route_turn(_route_state("I want to refresh knowledge graphs", _pending_model()))
    route = out.get("route", {})
    assert route.get("intent") == "pending_switch", route
    assert route.get("pause_pending") is True, route
    assert route.get("target_topic") == "Knowledge Graphs", route


def test_route_turn_pending_progress() -> None:
    with _patched_llm(["{}"]):
        out = tutor_agent.route_turn(_route_state("show my progress", _pending_model()))
    route = out.get("route", {})
    assert route.get("intent") == "progress", route


def test_merge_concept_updates_uses_update_topic() -> None:
    model = ensure_model_shape(default_student_model())
    updates = [
        {
            "topic": "Self Attention",
            "concept_id": "attention_mechanism",
            "label": "Attention Mechanism",
            "proficiency_delta": 0.08,
        },
        {
            "topic": "Knowledge Graphs",
            "concept_id": "neo4j",
            "label": "Neo4j",
            "proficiency_delta": 0.06,
        },
    ]
    merged = tutor_agent._merge_concept_updates(model=model, updates=updates, current_topic="General")
    topics = merged.get("topics", {})
    assert "Self Attention" in topics, topics
    assert "Knowledge Graphs" in topics, topics
    assert "attention_mechanism" in topics["Self Attention"]["concepts"], topics
    assert "neo4j" in topics["Knowledge Graphs"]["concepts"], topics


def test_pending_social_reanchors() -> None:
    model = _pending_model()
    state = _route_state("how are you", model)
    with _patched_llm(["{}"]):
        routed = tutor_agent.route_turn(state)
    with _patched_llm([]):
        out = tutor_agent.respond_turn({**state, **routed})
    messages = out.get("messages") or []
    last = messages[-1].content if messages else ""
    assert "continue our quick check" in last.lower(), last
    assert "pending question" in last.lower(), last


def test_pending_switch_pauses_and_continues_new_topic() -> None:
    state = _route_state("I want to refresh knowledge graphs", _pending_model())
    planner = json.dumps(
        {
            "next_action": "chat",
            "action_reason": "switch topic requested",
            "goal_concept": "",
            "interrupt_for_review": False,
            "should_gate": False,
            "gate_candidate": {},
        }
    )
    with _patched_llm(["{}", planner, "Great, let's refresh knowledge graphs starting from nodes and edges."]):
        routed = tutor_agent.route_turn(state)
        out = tutor_agent.respond_turn({**state, **routed})

    model = ensure_model_shape(out.get("student_model") or {})
    paused = model.get("paused_reviews", [])
    assert isinstance(paused, list) and len(paused) >= 1, model
    assert not model.get("active_review"), model
    messages = out.get("messages") or []
    last = messages[-1].content if messages else ""
    assert "knowledge graphs" in last.lower(), last


def test_pending_inability_teach_then_reask() -> None:
    state = _route_state("I can't", _pending_model())
    teach_reply = (
        "Attention helps a model focus on relevant tokens when processing a sequence.\n\n"
        "Quick check:\nHow does attention improve handling of long-range dependencies?"
    )
    with _patched_llm(["{}", teach_reply]):
        routed = tutor_agent.route_turn(state)
        out = tutor_agent.respond_turn({**state, **routed})

    messages = out.get("messages") or []
    last = messages[-1].content if messages else ""
    model = ensure_model_shape(out.get("student_model") or {})
    active = model.get("active_review", {})
    assert "quick check" in last.lower(), last
    assert active.get("status") == "awaiting_answer", active
    assert "how does attention improve" in str(active.get("question", "")).lower(), active


def test_update_student_model_seeds_review_with_update_topic() -> None:
    model = ensure_model_shape(default_student_model())
    state = {
        "student_model": model,
        "current_topic": "Self Attention",
        "concept_updates": [
            {
                "topic": "Knowledge Graphs",
                "concept_id": "neo4j",
                "label": "Neo4j",
                "proficiency_delta": 0.09,
                "evidence": {
                    "kind": "mentioned",
                    "text": "asked about neo4j",
                    "source": "student",
                },
            }
        ],
        "review_result": {},
        "consumed_student_text": "what is neo4j",
        "messages": [],
        "route": {
            "intent": "learn_request",
            "reason": "student asked to learn topic",
            "confidence": 0.9,
            "pause_pending": False,
        },
        "action_reason": "student asked to learn topic",
    }
    with _patched_store_save():
        out = tutor_agent.update_student_model(state, config={"configurable": {"thread_id": "test_router_v2"}})

    updated = ensure_model_shape(out.get("student_model") or {})
    queue = updated.get("review_queue", [])
    assert any(str(item.get("topic")) == "Knowledge Graphs" for item in queue if isinstance(item, dict)), queue
    route_log = updated.get("meta", {}).get("route_log", [])
    assert route_log and route_log[-1].get("intent") == "learn_request", route_log


def test_plan_next_step_fresh_topic_prefers_explain_over_gate() -> None:
    model = ensure_model_shape(default_student_model())
    state = {
        "student_answer": "I want to learn about knowledge graphs",
        "current_topic": "Knowledge Graphs",
        "student_model": model,
    }
    planner_forced_gate = json.dumps(
        {
            "next_action": "ask_prereq_diagnostic",
            "action_reason": "possible gap",
            "goal_concept": "knowledge_graphs",
            "interrupt_for_review": False,
            "should_gate": True,
            "gate_candidate": {
                "concept_id": "graph_theory",
                "label": "Graph Theory",
                "question": "What are nodes and edges?",
                "confidence": 0.9,
                "estimated_proficiency": 0.1,
            },
        }
    )
    with _patched_llm([planner_forced_gate]):
        out = tutor_agent.plan_next_step(state)
    assert out.get("next") == "chat", out
    assert "explain first" in str(out.get("action_reason", "")).lower(), out


def test_pedagogical_plan_confusion_prefers_explain_then_probe() -> None:
    model = ensure_model_shape(default_student_model())
    state = {
        "student_answer": "I am confused about self-attention and where pooling fits.",
        "current_topic": "Self-Attention",
        "student_model": model,
        "route": {"intent": "learn_request", "target_topic": "Self-Attention"},
    }
    with _patched_llm(["{}"]):
        out = tutor_agent.pedagogical_plan(state)
    plan = out.get("pedagogical_plan", {})
    assert plan.get("plan_mode") == "explain_then_probe", plan


def test_assess_understanding_does_not_drift_topic_on_progress() -> None:
    model = ensure_model_shape(default_student_model())
    model = tutor_agent._merge_concept_updates(
        model=model,
        updates=[
            {"topic": "Large Language Models", "concept_id": "large_language_models", "label": "Large Language Models", "proficiency_delta": 0.1},
            {"topic": "Self-Attention", "concept_id": "self_attention", "label": "Self-Attention", "proficiency_delta": 0.1},
        ],
        current_topic="General",
    )
    state = {
        "student_model": model,
        "current_topic": "Large Language Models",
        "consumed_student_text": "what does my student model look like now",
        "messages": tutor_agent._append_turn_messages(
            "what does my student model look like now",
            "Topic: Large Language Models\nSTRUGGLING ...",
        ),
        "route": {"intent": "progress", "target_topic": ""},
        "pedagogical_plan": {"plan_mode": "direct_answer"},
        "concept_updates": [],
    }
    assessment_json = json.dumps(
        {
            "understanding": "unknown",
            "misconception": "",
            "topic": "Self-Attention",
            "tutor_hint": "keep them anchored",
            "concept_updates": [],
        }
    )
    with _patched_llm([assessment_json]):
        out = tutor_agent.assess_understanding(state)
    assert out.get("current_topic") == "Large Language Models", out


def test_assess_understanding_ignores_progress_concept_writes() -> None:
    model = ensure_model_shape(default_student_model())
    state = {
        "student_model": model,
        "current_topic": "Self Attention",
        "consumed_student_text": "what does my student model look like now",
        "messages": tutor_agent._append_turn_messages(
            "what does my student model look like now",
            "CROSS-TOPIC STRUGGLING ...",
        ),
        "route": {"intent": "progress", "target_topic": ""},
        "pedagogical_plan": {"plan_mode": "direct_answer"},
        "concept_updates": [],
    }
    with _patched_llm(
        [
            json.dumps(
                {
                    "understanding": "partial",
                    "misconception": "fake",
                    "topic": "Knowledge Graphs",
                    "tutor_hint": "fake hint",
                    "concept_updates": [
                        {
                            "topic": "Knowledge Graphs",
                            "concept_id": "knowledge_graph",
                            "label": "Knowledge Graph",
                            "proficiency_delta": 0.2,
                            "evidence": {"kind": "mentioned", "text": "fake", "source": "student"},
                        }
                    ],
                }
            )
        ]
    ):
        out = tutor_agent.assess_understanding(state)
    assert out.get("current_topic") == "Self Attention", out
    assert out.get("concept_updates") == [], out


def test_progress_report_defaults_to_cross_topic() -> None:
    model = ensure_model_shape(default_student_model())
    model = tutor_agent._merge_concept_updates(
        model=model,
        updates=[
            {"topic": "Self Attention", "concept_id": "self_attention", "label": "Self Attention", "proficiency_delta": 0.12},
            {"topic": "Knowledge Graphs", "concept_id": "knowledge_graph", "label": "Knowledge Graph", "proficiency_delta": 0.08},
        ],
        current_topic="General",
    )
    state = {
        "student_model": model,
        "current_topic": "Self Attention",
        "student_answer": "show my progress",
    }
    out = tutor_agent.progress_report(state)
    messages = out.get("messages") or []
    last = messages[-1].content if messages else ""
    assert "CROSS-TOPIC" in last, last
    assert "Self Attention" in last and "Knowledge Graphs" in last, last


def test_assessment_reanchors_unknown_update_topics_to_target_topic() -> None:
    model = ensure_model_shape(default_student_model())
    state = {
        "student_model": model,
        "current_topic": "Self Attention",
        "consumed_student_text": "I am confused about self-attention and pooling",
        "messages": tutor_agent._append_turn_messages(
            "I am confused about self-attention and pooling",
            "Let's separate core attention from pooling.",
        ),
        "route": {"intent": "learn_request", "target_topic": "Self Attention"},
        "pedagogical_plan": {"plan_mode": "explain_then_probe"},
        "concept_updates": [],
    }
    with _patched_llm(
        [
            json.dumps(
                {
                    "understanding": "partial",
                    "misconception": "pooling is inside core self-attention",
                    "topic": "Self Attention",
                    "tutor_hint": "keep it concrete",
                    "concept_updates": [
                        {
                            "topic": "Self Attention Mechanism",
                            "concept_id": "mean_max_pooling",
                            "label": "mean/max pooling",
                            "proficiency_delta": 0.06,
                            "evidence": {"kind": "mentioned", "text": "mean/max pooling", "source": "student"},
                        }
                    ],
                }
            )
        ]
    ):
        out = tutor_agent.assess_understanding(state)

    updates = out.get("concept_updates") or []
    assert updates, out
    assert all(str(item.get("topic")) == "Self Attention" for item in updates if isinstance(item, dict)), updates


def test_filter_maps_near_duplicate_concept_to_existing() -> None:
    model = ensure_model_shape(default_student_model())
    model = tutor_agent._merge_concept_updates(
        model=model,
        updates=[
            {
                "topic": "Self Attention",
                "concept_id": "self_attention",
                "label": "Self-Attention",
                "proficiency_delta": 0.12,
                "evidence": {"kind": "introduced", "text": "seed", "source": "tutor"},
            }
        ],
        current_topic="Self Attention",
    )
    filtered = tutor_agent._filter_concept_updates_for_persistence(
        model=model,
        updates=[
            {
                "topic": "Self Attention",
                "concept_id": "self_attention_mechanism",
                "label": "Self Attention Mechanism",
                "proficiency_delta": 0.04,
                "evidence": {"kind": "mentioned", "text": "mention", "source": "student"},
            }
        ],
        route_intent="learn_request",
        default_topic="Self Attention",
    )
    assert filtered, filtered
    assert filtered[0].get("concept_id") == "self_attention", filtered
    assert str(filtered[0].get("label", "")).lower().startswith("self-attention"), filtered


def test_update_student_model_merges_near_duplicate_updates() -> None:
    model = ensure_model_shape(default_student_model())
    model = tutor_agent._merge_concept_updates(
        model=model,
        updates=[
            {
                "topic": "Self Attention",
                "concept_id": "self_attention",
                "label": "Self-Attention",
                "proficiency_delta": 0.1,
                "evidence": {"kind": "introduced", "text": "seed", "source": "tutor"},
            }
        ],
        current_topic="Self Attention",
    )
    state = {
        "student_model": model,
        "current_topic": "Self Attention",
        "concept_updates": [
            {
                "topic": "Self Attention",
                "concept_id": "self_attention_mechanism",
                "label": "Self Attention Mechanism",
                "proficiency_delta": 0.04,
                "evidence": {"kind": "mentioned", "text": "student mention", "source": "student"},
            }
        ],
        "review_result": {},
        "consumed_student_text": "I think I get self-attention mechanism",
        "messages": [],
        "route": {
            "intent": "learn_request",
            "reason": "student asked to learn topic",
            "confidence": 0.9,
            "pause_pending": False,
        },
        "action_reason": "student asked to learn topic",
    }
    with _patched_store_save():
        out = tutor_agent.update_student_model(state, config={"configurable": {"thread_id": "test_router_v2"}})

    updated = ensure_model_shape(out.get("student_model") or {})
    topics = updated.get("topics", {})
    topic_data = topics.get("Self Attention", {})
    concepts = topic_data.get("concepts", {}) if isinstance(topic_data, dict) else {}
    assert "self_attention" in concepts, concepts
    assert "self_attention_mechanism" not in concepts, concepts


def main() -> None:
    tests = [
        test_route_turn_pending_social,
        test_route_turn_pending_inability,
        test_route_turn_pending_switch,
        test_route_turn_pending_progress,
        test_merge_concept_updates_uses_update_topic,
        test_pending_social_reanchors,
        test_pending_switch_pauses_and_continues_new_topic,
        test_pending_inability_teach_then_reask,
        test_update_student_model_seeds_review_with_update_topic,
        test_plan_next_step_fresh_topic_prefers_explain_over_gate,
        test_pedagogical_plan_confusion_prefers_explain_then_probe,
        test_assess_understanding_does_not_drift_topic_on_progress,
        test_assess_understanding_ignores_progress_concept_writes,
        test_progress_report_defaults_to_cross_topic,
        test_assessment_reanchors_unknown_update_topics_to_target_topic,
        test_filter_maps_near_duplicate_concept_to_existing,
        test_update_student_model_merges_near_duplicate_updates,
    ]
    for test_fn in tests:
        test_fn()
    print("test_tutor_router_v2.py: PASS")


if __name__ == "__main__":
    main()
