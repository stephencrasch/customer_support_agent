"""Adaptive tutor graph with persistent knowledge and scheduled review.

Key design choices:
- Minimal DAG with LLM-first turn routing and response behavior.
- Deterministic state transitions only for memory safety and persistence.
- Conservative concept updates with topic-scoped merging and review scheduling.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from student_store import (
    canonicalize_topic_label,
    ensure_model_shape,
    get_due_reviews,
    load_student_model as store_load_student_model,
    model_snapshot,
    review_snapshot,
    save_student_model as store_save_student_model,
    update_review_queue,
)

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
TUTOR_ROUTER_V2 = os.getenv("TUTOR_ROUTER_V2", "true").strip().lower() not in {"0", "false", "no", "off"}


class TutorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_topic: str
    student_answer: str
    consumed_student_text: str
    student_model: Dict[str, Any]
    tutor_hint: str
    concept_updates: List[Dict[str, Any]]
    review_result: Dict[str, Any]
    planned_review_target: Dict[str, Any]
    planned_goal_concept: str
    route: Dict[str, Any]
    pedagogical_plan: Dict[str, Any]
    understanding_assessment: Dict[str, Any]
    response_mode: str
    pending_context_message: str
    next: str
    action_reason: str


def extract_json(content: str) -> dict:
    content = (content or "").strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]
    return json.loads(content.strip())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _norm_alnum(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())


def _to_concept_id(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    concept_id = "".join(out).strip("_")
    return concept_id or "concept"


def _to_optional_concept_id(s: str) -> str:
    raw = (s or "").strip()
    if not raw:
        return ""
    if not any(ch.isalnum() for ch in raw):
        return ""
    concept_id = _to_concept_id(raw)
    return concept_id


def _clamp01(x: Any) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _collect_concept_labels(model: Dict[str, Any]) -> Dict[str, List[str]]:
    direct_concepts = model.get("concepts")
    topics = model.get("topics")

    concept_rows: List[tuple[str, Dict[str, Any]]] = []
    if isinstance(direct_concepts, dict):
        for concept_id, meta in direct_concepts.items():
            if isinstance(concept_id, str) and isinstance(meta, dict):
                concept_rows.append((concept_id, meta))

    if isinstance(topics, dict):
        for topic_data in topics.values():
            if not isinstance(topic_data, dict):
                continue
            topic_concepts = topic_data.get("concepts")
            if not isinstance(topic_concepts, dict):
                continue
            for concept_id, meta in topic_concepts.items():
                if isinstance(concept_id, str) and isinstance(meta, dict):
                    concept_rows.append((concept_id, meta))

    if not concept_rows:
        return {}

    out: Dict[str, List[str]] = {}
    for concept_id, meta in concept_rows:
        names = out.get(concept_id, [])
        label = meta.get("label")
        if isinstance(label, str) and label.strip():
            names.append(_norm_text(label))
        aliases = meta.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    names.append(_norm_text(alias))
        names.append(_norm_text(concept_id))

        seen = set()
        uniq: List[str] = []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                uniq.append(name)
        out[concept_id] = uniq
    return out


def _concepts_mentioned_in_text(model: Dict[str, Any], text: str) -> List[str]:
    haystack = _norm_text(text)
    if not haystack:
        return []

    labels = _collect_concept_labels(model)
    mentioned: List[str] = []
    for concept_id, names in labels.items():
        if any(name and name in haystack for name in names):
            mentioned.append(concept_id)
    return mentioned


def _last_user_and_ai(messages: List[BaseMessage]) -> tuple[str, str]:
    last_user = ""
    last_ai = ""
    for message in reversed(messages or []):
        if not last_ai and isinstance(message, AIMessage):
            last_ai = message.content
        if not last_user and isinstance(message, HumanMessage):
            last_user = message.content
        if last_user and last_ai:
            break
    return last_user, last_ai


def _append_turn_messages(student_text: str, ai_text: str) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    if student_text:
        out.append(HumanMessage(content=student_text))
    out.append(AIMessage(content=ai_text))
    return out


def _concept_proficiency(model: Dict[str, Any], topic: str, concept_id: str) -> float:
    topics = model.get("topics")
    if not isinstance(topics, dict):
        return 0.0
    topic_data = topics.get(topic)
    if not isinstance(topic_data, dict):
        return 0.0
    concepts = topic_data.get("concepts")
    if not isinstance(concepts, dict):
        return 0.0
    meta = concepts.get(concept_id)
    if not isinstance(meta, dict):
        return 0.0
    return _clamp01(meta.get("proficiency", 0.0))


def _quality_from_score(score: int) -> int:
    if score >= 90:
        return 5
    if score >= 75:
        return 4
    if score >= 55:
        return 3
    if score >= 35:
        return 2
    return 1


def _delta_from_score(score: int) -> float:
    if score >= 90:
        return 0.22
    if score >= 75:
        return 0.14
    if score >= 55:
        return 0.06
    if score >= 35:
        return -0.02
    return -0.12


def _quality_from_delta(delta: float) -> int:
    if delta >= 0.18:
        return 5
    if delta >= 0.1:
        return 4
    if delta >= 0.03:
        return 3
    if delta >= -0.05:
        return 2
    return 1


def _minimum_score_from_overlap(*, label: str, question: str, answer: str) -> int:
    lowered = _norm_text(answer)
    if not lowered:
        return 0

    tokens: List[str] = []
    for raw in (label, question):
        for token in _norm_text(raw).split():
            if len(token) >= 4 and token not in {"quick", "check", "before", "about", "neural", "networks"}:
                tokens.append(token)
    unique_tokens = list(dict.fromkeys(tokens))
    if not unique_tokens:
        return 0

    matches = sum(1 for token in unique_tokens if token in lowered)
    if matches <= 0:
        return 0
    if matches == 1:
        return 30
    if matches == 2:
        return 45
    return 60


def _extract_quick_check_question(text: str) -> str:
    body = (text or "").strip()
    if not body or "?" not in body:
        return ""
    q_idx = body.rfind("?")
    start_newline = body.rfind("\n", 0, q_idx)
    start_sentence = body.rfind(". ", 0, q_idx)
    start = max(start_newline, start_sentence)
    if start == -1:
        start = 0
    elif start == start_sentence:
        start += 2
    else:
        start += 1
    candidate = " ".join(body[start : q_idx + 1].split())
    if len(candidate) < 5 or len(candidate) > 240:
        return ""
    return candidate


def _best_proficiency_for_concept(model: Dict[str, Any], concept_id: str, label: str) -> float:
    topics = model.get("topics")
    if not isinstance(topics, dict):
        return 0.0

    wanted_id = _norm_text(concept_id)
    wanted_label = _norm_text(label)
    best = 0.0
    for topic_data in topics.values():
        if not isinstance(topic_data, dict):
            continue
        concepts = topic_data.get("concepts")
        if not isinstance(concepts, dict):
            continue
        for existing_id, meta in concepts.items():
            if not isinstance(existing_id, str) or not isinstance(meta, dict):
                continue
            existing_label = meta.get("label", existing_id)
            if _norm_text(existing_id) != wanted_id and _norm_text(str(existing_label)) != wanted_label:
                continue
            best = max(best, _clamp01(meta.get("proficiency", 0.0)))
    return best


def _sanitize_gate_candidate(raw: Dict[str, Any], *, current_topic: str, goal_concept: str) -> Dict[str, Any]:
    concept_id_raw = str(raw.get("concept_id") or raw.get("label") or "").strip()
    if not concept_id_raw:
        return {}
    concept_id = _to_concept_id(concept_id_raw)
    label_raw = str(raw.get("label") or concept_id.replace("_", " ").title()).strip()
    label = label_raw or concept_id.replace("_", " ").title()
    question = str(raw.get("question") or "").strip()
    if not question:
        question = f"Quick check before we continue: can you explain {label} in your own words?"

    confidence = _clamp01(raw.get("confidence", 0.0))
    estimated_proficiency = _clamp01(raw.get("estimated_proficiency", 0.0))

    topic = str(raw.get("topic") or current_topic or goal_concept.replace("_", " ").title() or "General").strip()
    if not topic:
        topic = "General"

    return {
        "topic": topic,
        "concept_id": concept_id,
        "label": label,
        "question": question,
        "confidence": confidence,
        "estimated_proficiency": estimated_proficiency,
        "goal_concept": goal_concept,
    }


def _fallback_delta_for_update(update: Dict[str, Any]) -> float:
    try:
        parsed = float(update.get("proficiency_delta", 0.0))
    except (TypeError, ValueError):
        parsed = 0.0
    if abs(parsed) >= 0.01:
        return parsed

    evidence = update.get("evidence")
    kind = ""
    if isinstance(evidence, dict):
        kind = str(evidence.get("kind") or "").strip().lower()

    if kind in {"correct_explanation", "correct_application"}:
        return 0.12
    if kind in {"asked"}:
        return 0.04
    if kind in {"introduced"}:
        return 0.03
    if kind in {"mentioned"}:
        return 0.02
    if kind in {"confused", "incorrect"}:
        return -0.1
    return 0.0


def _extract_llm_concept_updates(
    *,
    model: Dict[str, Any],
    student_text: str,
    current_topic: str,
    existing_updates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    message = (student_text or "").strip()
    if not message:
        return []

    existing_ids = set()
    for update in existing_updates:
        if not isinstance(update, dict):
            continue
        raw_id = str(update.get("concept_id") or "").strip()
        raw_label = str(update.get("label") or "").strip()
        concept_id = _to_optional_concept_id(raw_id) or _to_optional_concept_id(raw_label)
        if concept_id:
            existing_ids.add(concept_id)

    known_labels = _collect_concept_labels(model)
    known_concepts: List[Dict[str, Any]] = []
    for concept_id, aliases in known_labels.items():
        alias_list = [alias for alias in aliases if isinstance(alias, str) and alias.strip()]
        if not alias_list:
            continue
        known_concepts.append({"concept_id": concept_id, "aliases": alias_list[:4]})
    known_concepts = known_concepts[:40]

    prompt = CONCEPT_MENTION_EXTRACTOR_PROMPT.format(
        student_message=message,
        current_topic=current_topic or "General",
        known_concepts_json=json.dumps(known_concepts, ensure_ascii=False, indent=2, sort_keys=True),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {}

    raw_candidates = data.get("concept_candidates")
    if not isinstance(raw_candidates, list):
        return []

    updates: List[Dict[str, Any]] = []
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        confidence = _clamp01(item.get("confidence", 0.0))
        if confidence < 0.62:
            continue
        raw_id = str(item.get("concept_id") or "").strip()
        raw_label = str(item.get("label") or "").strip()
        concept_id = _to_optional_concept_id(raw_id) or _to_optional_concept_id(raw_label)
        if not concept_id or concept_id in existing_ids:
            continue
        label = raw_label or concept_id.replace("_", " ").title()
        evidence_text = str(item.get("evidence_text") or message).strip()
        if not evidence_text:
            evidence_text = message
        updates.append(
            {
                "topic": (current_topic or "General").strip() or "General",
                "concept_id": concept_id,
                "label": label,
                "proficiency_delta": 0.02,
                "evidence": {
                    "kind": "mentioned",
                    "text": evidence_text[:180],
                    "source": "student",
                    "reasoning": f"llm concept extractor confidence={confidence:.2f}",
                },
            }
        )
        existing_ids.add(concept_id)
        if len(updates) >= 4:
            break

    return updates


def _pick_review_target(model: Dict[str, Any], current_topic: str) -> dict[str, Any] | None:
    due = get_due_reviews(model, current_topic=current_topic, limit=1)
    if due:
        return due[0]

    topics = model.get("topics")
    if not isinstance(topics, dict):
        return None

    best: dict[str, Any] | None = None
    for topic_name, topic_data in topics.items():
        if not isinstance(topic_data, dict):
            continue
        concepts = topic_data.get("concepts")
        if not isinstance(concepts, dict):
            continue
        for concept_id, meta in concepts.items():
            if not isinstance(concept_id, str) or not isinstance(meta, dict):
                continue
            try:
                proficiency = float(meta.get("proficiency", 0.0))
            except (TypeError, ValueError):
                proficiency = 0.0
            candidate = {
                "topic": topic_name,
                "concept_id": concept_id,
                "label": meta.get("label", concept_id),
                "proficiency": _clamp01(proficiency),
            }
            if best is None or candidate["proficiency"] < best["proficiency"]:
                best = candidate
    return best


def _find_existing_concept_target(model: Dict[str, Any], *, concept_id: str, label: str) -> Dict[str, Any] | None:
    topics = model.get("topics")
    if not isinstance(topics, dict):
        return None

    wanted_id = _norm_text(concept_id)
    wanted_label = _norm_text(label)
    best: Dict[str, Any] | None = None
    for topic_name, topic_data in topics.items():
        if not isinstance(topic_data, dict):
            continue
        concepts = topic_data.get("concepts")
        if not isinstance(concepts, dict):
            continue
        for existing_id, meta in concepts.items():
            if not isinstance(existing_id, str) or not isinstance(meta, dict):
                continue
            existing_label = str(meta.get("label") or existing_id)
            if _norm_text(existing_id) != wanted_id and _norm_text(existing_label) != wanted_label:
                continue
            proficiency = _clamp01(meta.get("proficiency", 0.0))
            candidate = {
                "topic": str(topic_name or "General"),
                "concept_id": existing_id,
                "label": existing_label,
                "proficiency": proficiency,
            }
            if best is None or candidate["proficiency"] < best["proficiency"]:
                best = candidate
    return best


_MULTI_CONCEPT_SPLIT_RE = re.compile(r"\b(?:and|or|plus|then|also|vs|versus|with)\b|[,;/]", re.IGNORECASE)


def _first_concept_fragment(text: str) -> str:
    phrase = (text or "").strip()
    if not phrase:
        return ""
    head = _MULTI_CONCEPT_SPLIT_RE.split(phrase, maxsplit=1)[0].strip()
    return head.strip(" \t\n\r.,!?;:\"'()[]{}")


def _extract_review_phrase(student_text: str) -> str:
    lowered = _norm_text(student_text)
    if not lowered:
        return ""
    markers = ("review", "quiz me on", "test me on", "practice")
    phrase = ""
    for marker in markers:
        if marker in lowered:
            after = lowered.split(marker, 1)[1].strip()
            if after:
                phrase = after
                break
    if not phrase:
        return ""

    prefixes = ("about ", "on ", "the ", "my ", "a ", "an ")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if phrase.startswith(prefix):
                phrase = phrase[len(prefix) :].strip()
                changed = True
    phrase = phrase.strip(" \t\n\r.,!?;:\"'()[]{}")
    if not phrase:
        return ""

    tokens = phrase.split()
    filler = {
        "today",
        "tonight",
        "tomorrow",
        "now",
        "later",
        "please",
        "again",
        "currently",
        "right",
        "away",
    }
    while tokens and _norm_alnum(tokens[0]) in filler:
        tokens.pop(0)
    while tokens and _norm_alnum(tokens[-1]) in filler:
        tokens.pop()
    if not tokens:
        return ""
    return _first_concept_fragment(" ".join(tokens).strip())


def _infer_requested_review_target(
    *,
    student_text: str,
    model: Dict[str, Any],
    current_topic: str,
) -> Dict[str, Any] | None:
    extracted = _extract_llm_concept_updates(
        model=model,
        student_text=student_text,
        current_topic=current_topic,
        existing_updates=[],
    )
    raw_id = ""
    raw_label = ""
    if extracted:
        first = extracted[0]
        raw_id = str(first.get("concept_id") or "").strip()
        raw_label = str(first.get("label") or "").strip()
    else:
        raw_label = _extract_review_phrase(student_text)

    normalized_label = _first_concept_fragment(raw_label)
    normalized_id_label = _first_concept_fragment(raw_id.replace("_", " "))
    concept_id = (
        _to_optional_concept_id(normalized_label)
        or _to_optional_concept_id(normalized_id_label)
        or _to_optional_concept_id(raw_id)
        or _to_optional_concept_id(raw_label)
    )
    if not concept_id or not _is_specific_goal_concept(concept_id):
        return None
    label = normalized_label or normalized_id_label or raw_label.strip() or concept_id.replace("_", " ").title()

    existing = _find_existing_concept_target(model, concept_id=concept_id, label=label)
    if existing:
        return existing

    topic_guess = (current_topic or "").strip()
    if not topic_guess:
        topic_guess = "General"

    return {
        "topic": topic_guess,
        "concept_id": concept_id,
        "label": label,
        "proficiency": 0.0,
    }


def _has_review_item(model: Dict[str, Any], concept_id: str, topic: str = "") -> bool:
    queue = model.get("review_queue")
    if not isinstance(queue, list):
        return False
    topic_norm = (topic or "").strip().lower()
    for item in queue:
        if not isinstance(item, dict):
            continue
        if str(item.get("concept_id") or "") != concept_id:
            continue
        item_topic = str(item.get("topic") or "").strip().lower()
        if topic_norm and item_topic != topic_norm:
            continue
        if not topic_norm or item_topic == topic_norm:
            return True
    return False


def _merge_concept_updates(
    *,
    model: Dict[str, Any],
    updates: List[Dict[str, Any]],
    current_topic: str,
    evidence_per_concept: int = 12,
) -> Dict[str, Any]:
    model = ensure_model_shape(model)
    topics = model.get("topics", {})
    if not isinstance(topics, dict):
        topics = {}

    now = _utc_now_iso()

    for update in updates or []:
        if not isinstance(update, dict):
            continue

        label = update.get("label")
        if not isinstance(label, str) or not label.strip():
            continue

        topic = canonicalize_topic_label(str(update.get("topic") or current_topic or "General"))
        topic_data = topics.get(topic, {})
        if not isinstance(topic_data, dict):
            topic_data = {"concepts": {}, "last_studied": None, "total_practice": 0}

        concepts = topic_data.get("concepts", {})
        if not isinstance(concepts, dict):
            concepts = {}

        labels = _collect_concept_labels({"concepts": concepts})
        name_to_id: Dict[str, str] = {}
        for concept_id, names in labels.items():
            for name in names:
                if name and name not in name_to_id:
                    name_to_id[name] = concept_id

        requested_id = update.get("concept_id")
        guess_id = requested_id if isinstance(requested_id, str) and requested_id.strip() else _to_concept_id(label)
        guess_id = _to_concept_id(guess_id)

        target_id = name_to_id.get(_norm_text(label)) or name_to_id.get(_norm_text(guess_id))
        if not target_id:
            aliases = update.get("aliases")
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip():
                        target_id = name_to_id.get(_norm_text(alias))
                        if target_id:
                            break
        if not target_id:
            target_id = guess_id

        meta = concepts.get(target_id)
        is_new = not isinstance(meta, dict)
        if not isinstance(meta, dict):
            meta = {}

        existing_label = meta.get("label")
        if isinstance(existing_label, str) and existing_label.strip():
            if existing_label.strip() != label.strip():
                aliases = meta.get("aliases")
                if not isinstance(aliases, list):
                    aliases = []
                if label.strip() not in aliases:
                    aliases.append(label.strip())
                meta["aliases"] = aliases
        else:
            meta["label"] = label.strip()

        incoming_aliases = update.get("aliases")
        if isinstance(incoming_aliases, list):
            aliases = meta.get("aliases")
            if not isinstance(aliases, list):
                aliases = []
            for alias in incoming_aliases:
                if isinstance(alias, str) and alias.strip() and alias.strip() not in aliases:
                    aliases.append(alias.strip())
            meta["aliases"] = aliases

        old_prof = _clamp01(meta.get("proficiency", 0.0))
        try:
            delta = float(update.get("proficiency_delta", 0.0))
        except (TypeError, ValueError):
            delta = 0.0
        new_prof = _clamp01(old_prof + delta)
        meta["proficiency"] = new_prof

        mastery = update.get("mastery_level")
        if mastery in {"introduced", "developing", "proficient", "mastered"}:
            meta["mastery_level"] = mastery
        elif new_prof < 0.3:
            meta["mastery_level"] = "introduced"
        elif new_prof < 0.6:
            meta["mastery_level"] = "developing"
        elif new_prof < 0.85:
            meta["mastery_level"] = "proficient"
        else:
            meta["mastery_level"] = "mastered"

        if delta > 0.05:
            meta["trajectory"] = "improving"
        elif delta < -0.05:
            meta["trajectory"] = "declining"
        else:
            meta["trajectory"] = "stable"

        if is_new:
            meta["first_seen"] = now
        meta["last_practiced"] = now
        try:
            count = int(meta.get("practice_count", 0))
        except (TypeError, ValueError):
            count = 0
        meta["practice_count"] = count + 1

        notes = update.get("notes")
        if isinstance(notes, str) and notes.strip():
            meta["notes"] = notes.strip()

        evidence = update.get("evidence")
        if isinstance(evidence, dict):
            history = meta.get("evidence")
            if not isinstance(history, list):
                history = []
            event = {
                "kind": evidence.get("kind", "observation"),
                "text": evidence.get("text", ""),
                "source": evidence.get("source", "turn"),
                "timestamp": now,
                "proficiency_before": old_prof,
                "proficiency_after": new_prof,
                "delta": delta,
            }
            if "reasoning" in evidence:
                event["reasoning"] = evidence.get("reasoning", "")
            history.append(event)
            meta["evidence"] = history[-evidence_per_concept:]

        concepts[target_id] = meta
        topic_data["concepts"] = concepts
        topic_data["last_studied"] = now
        try:
            practice_total = int(topic_data.get("total_practice", 0))
        except (TypeError, ValueError):
            practice_total = 0
        topic_data["total_practice"] = practice_total + 1
        topics[topic] = topic_data

    model["topics"] = topics
    return model


WATCHER_PROMPT = """You are an internal learning observer for a tutoring system.

Previous topic: {current_topic}

Student last message:
{student_message}

Tutor last reply:
{tutor_reply}

Known related concepts in this topic:
{related_concepts_json}

Return only JSON with:
{{
  "current_topic": "Topic name",
  "tutor_hint": "1-2 internal coaching sentences",
  "concept_updates": [
    {{
      "concept_id": "snake_case",
      "label": "Concept name",
      "aliases": ["optional"],
      "proficiency_delta": 0.0,
      "mastery_level": "introduced|developing|proficient|mastered",
      "notes": "optional",
      "evidence": {{
        "kind": "confused|asked|correct_explanation|incorrect|introduced|observation",
        "text": "short quote/paraphrase",
        "source": "student|tutor",
        "reasoning": "why this delta"
      }}
    }}
  ]
}}

Constraints:
- Keep updates conservative.
- Prefer existing concepts when possible.
- Update at most 3 concepts.
- If a concept is explicitly mentioned in this turn and missing from known concepts, add it with small delta.
"""


CHAT_PROMPT = """You are an adaptive tutor.

Current topic: {current_topic}

Student proficiency snapshot:
{student_snapshot}

Review status:
{review_status}

Recent internal observation:
{tutor_hint}

Instructions:
1. Address the student's exact question first, in plain language.
2. If they express confusion, identify the misconception and correct it gently.
3. Prefer conceptual next steps (a worked example, analogy, or quick check question).
4. Do not suggest tools/framework code (PyTorch/TensorFlow), papers, or external resources unless the student explicitly asks for implementation/resources.
5. Keep the tone natural and conversational (like an in-person tutor), not scripted.
6. Separate core mechanism from optional add-ons when the student mixes concepts (for example: explain that vanilla self-attention is Q/K/V + softmax-weighted sum; pooling is a downstream aggregation choice).
7. Keep it concise (2-4 short paragraphs).
"""


REVIEW_QUESTION_PROMPT = """You are generating one concise review question.

Topic: {topic}
Concept: {concept_label}
Proficiency: {proficiency}

Return only the question text.
The question should check understanding, not trivia.
"""


REVIEW_GRADER_PROMPT = """Grade a student's answer to a review question.

Topic: {topic}
Concept: {concept_label}
Question: {question}
Student answer: {student_answer}

Return JSON only:
{{
  "score": 0,
  "feedback": "1-2 sentence feedback"
}}
"""


FOUNDATION_TEACH_PROMPT = """You are a patient tutor teaching a foundational concept.

Concept: {concept_label}
Why it matters for next goal: {goal_concept}
Current topic context: {current_topic}
Student message: {student_message}

Write a concise explanation with:
1) simple definition
2) one concrete, domain-relevant example
3) one quick check question at the end

Teaching style rules:
- Keep the explanation conversational, not rigidly templated.
- Use neural-network examples only when they are relevant to the concept/topic.
- If the student asked about a different concept than the pending one, acknowledge the switch briefly.
"""


PREREQ_PLANNER_PROMPT = """You are planning the next tutoring step.

Student message:
{student_message}

Current topic:
{current_topic}

Progress snapshot:
{progress_snapshot}

Review snapshot:
{review_snapshot}

Topic history:
{topic_history}

Return JSON only:
{{
  "next_action": "chat|ask_prereq_diagnostic|ask_review_question",
  "action_reason": "short reason",
  "goal_concept": "snake_case_or_empty",
  "interrupt_for_review": false,
  "should_gate": true,
  "gate_candidate": {{
    "concept_id": "snake_case_foundation_concept",
    "label": "Human-readable concept",
    "question": "One quick diagnostic question",
    "confidence": 0.0,
    "estimated_proficiency": 0.0
  }}
}}

Rules:
- Prefer chat when uncertain.
- For a fresh topic request, start with a concise high-level explanation first.
- Do not gate immediately on first-touch learning requests unless the student explicitly asks to be assessed.
- Use prereq gating only when there is a clear likely gap and the student is asking for depth/assessment.
- Set interrupt_for_review=true only if a review question now would feel natural.
- confidence is 0..1.
- estimated_proficiency is 0..1.
"""


INTENT_ROUTER_PROMPT = """Route the student's latest message to the best tutoring action.

Student message:
{student_message}

Current topic:
{current_topic}

Has active prerequisite context:
{prereq_context}

Current review snapshot:
{review_snapshot}

Return JSON only:
{{
  "next_action": "progress_report|ask_review_question|teach_foundation|plan_next_step",
  "reason": "short reason",
  "confidence": 0.0
}}

Rules:
- Choose progress_report for progress/knowledge-graph requests.
- Choose ask_review_question for explicit requests to review, quiz, or practice.
- Choose teach_foundation only when the student clearly asks to teach/explain and prereq_context is true.
- Otherwise choose plan_next_step.
"""


TURN_ROUTER_PROMPT = """Route the student's latest message for an adaptive tutor.

Student message:
{student_message}

Current topic:
{current_topic}

Has pending quick check:
{has_pending_review}

Pending quick-check question:
{pending_question}

Pending concept:
{pending_concept}

Paused review count:
{paused_count}

Current review snapshot:
{review_snapshot}

Return JSON only:
{{
  "intent": "session_start|pending_social|pending_answer|pending_defer|pending_switch|progress|review_request|learn_request|chat",
  "reason": "short reason",
  "confidence": 0.0,
  "pause_pending": false,
  "resume_pending": false,
  "target_topic": "optional topic string or empty"
}}

Rules:
- If there is a pending quick check and the student sends social small talk, choose pending_social.
- pending_social means: social response + ask whether to continue the pending quick check.
- If there is a pending quick check and student says defer/skip/not now, choose pending_defer.
- If there is a pending quick check and student clearly switches tasks/topics, choose pending_switch and set pause_pending=true.
- If there is a pending quick check and student attempts an answer (including \"I can't\" / \"I don't know\"), choose pending_answer.
- Choose progress for explicit progress/knowledge-graph requests.
- Choose review_request for explicit review/quiz/practice requests.
- Choose learn_request for explicit requests to learn/teach/explain a topic.
- Choose session_start only when student message is empty.
"""


PEDAGOGICAL_PLAN_PROMPT = """Plan the next tutoring move using instructional reasoning.

Student message:
{student_message}

Route intent:
{route_intent}

Current topic:
{current_topic}

Progress snapshot:
{progress_snapshot}

Review snapshot:
{review_snapshot}

Return JSON only:
{{
  "plan_mode": "direct_answer|explain_then_probe|probe_only|teach_gap_then_resume|social_reanchor",
  "target_topic": "optional topic string or empty",
  "target_concept": "optional concept label or empty",
  "misconception": "short text or empty",
  "response_instructions": "1-2 concise instructional moves",
  "suspected_gaps": [
    {{
      "concept_id": "snake_case_or_empty",
      "label": "Gap concept",
      "reason": "short rationale",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- First help with the student's exact question before redirecting.
- If student signals confusion, prefer explain_then_probe.
- Probe with one short comprehension check question, not a quiz.
- Avoid tool/framework/resource suggestions unless student asks for implementation/resources.
- suspected_gaps must be conservative and confidence in 0..1.
"""


UNDERSTANDING_ASSESSMENT_PROMPT = """Assess understanding from the latest tutor exchange.

Current topic:
{current_topic}

Route intent:
{route_intent}

Student message:
{student_message}

Tutor reply:
{tutor_reply}

Pedagogical plan:
{pedagogical_plan_json}

Known related concepts:
{related_concepts_json}

Return JSON only:
{{
  "understanding": "unknown|confused|partial|solid",
  "misconception": "short text or empty",
  "topic": "topic this turn was about",
  "tutor_hint": "one short internal coaching line",
  "concept_updates": [
    {{
      "topic": "Topic name",
      "concept_id": "snake_case",
      "label": "Concept name",
      "proficiency_delta": 0.0,
      "evidence": {{
        "kind": "confused|asked|correct_explanation|incorrect|introduced|observation|mentioned",
        "text": "short quote/paraphrase",
        "source": "student|tutor",
        "reasoning": "why this update"
      }}
    }}
  ]
}}

Rules:
- Keep updates conservative and topic-specific.
- If student confusion is explicit, include at least one confused/partial signal.
- Do not invent many updates; at most 4.
"""


PENDING_REVIEW_ROUTER_PROMPT = """Route the student's latest message while a review question is pending.

Student message:
{student_message}

Pending review question:
{pending_question}

Pending concept:
{pending_concept}

Current topic:
{current_topic}

Prerequisite-gate context:
{prereq_context}

Current review snapshot:
{review_snapshot}

Return JSON only:
{{
  "next_action": "grade_review_answer|teach_foundation|remind_review_question|progress_report|chat|plan_next_step",
  "reason": "short reason",
  "confidence": 0.0,
  "pause_pending_review": false
}}

Rules:
- Treat short attempts like "I don't know", "I can't", or partial answers as grade_review_answer (they are valid attempts).
- Choose teach_foundation when the student asks to be taught/explained before answering.
- Choose remind_review_question for defer/pause/skip requests about the pending question.
- Choose progress_report only for explicit progress/graph requests.
- Choose chat for social talk not tied to a task.
- Choose plan_next_step for explicit task switches or topic switches unrelated to the pending question.
- If you choose plan_next_step because the student switched tasks/topics, set pause_pending_review=true.
"""


CONCEPT_MENTION_EXTRACTOR_PROMPT = """Extract explicit learning concept mentions from the student message.

Student message:
{student_message}

Current topic:
{current_topic}

Known concepts:
{known_concepts_json}

Return JSON only:
{{
  "concept_candidates": [
    {{
      "concept_id": "snake_case_or_empty",
      "label": "Concept name",
      "confidence": 0.0,
      "evidence_text": "short phrase from student message"
    }}
  ]
}}

Rules:
- Include only concepts explicitly mentioned or strongly implied by the student.
- Prefer known concepts when they clearly match.
- If concept_id is unknown, leave it empty but provide label.
- confidence is 0..1.
- Include at most 5 candidates.
"""


PROGRESS_KEYWORDS = (
    "progress",
    "weakest",
    "show my graph",
    "my knowledge graph",
    "my knowlege graph",
    "what does my knowledge graph contain",
    "what are my topics",
    "current graph",
    "what do i know",
    "what concepts do i know",
    "where am i",
    "how am i doing",
)
REVIEW_KEYWORDS = (
    "review",
    "quiz me",
    "test me",
    "practice",
    "what should i review",
    "due",
)
DEFER_REVIEW_KEYWORDS = (
    "skip",
    "pass",
    "not now",
    "later",
    "defer",
    "come back",
    "move on",
    "dont want to answer",
    "don't want to answer",
    "no answer right now",
)
LEARN_INTENT_KEYWORDS = (
    "i want to learn",
    "learn about",
    "teach me",
    "help me learn",
    "can you teach",
    "refresh my memory",
    "refresh",
    "brush up on",
)
TEACH_INTENT_KEYWORDS = (
    "teach me",
    "explain",
    "walk me through",
    "step by step",
    "step-by-step",
)
SOCIAL_KEYWORDS = (
    "hey",
    "hi",
    "hello",
    "how are you",
    "hows it going",
    "how's it going",
    "whats up",
    "what's up",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
)


def _is_progress_request(text: str) -> bool:
    lowered = _norm_text(text)
    if any(keyword in lowered for keyword in PROGRESS_KEYWORDS):
        return True

    alnum = _norm_alnum(text)
    if "knowledgegraph" in alnum or "knowlegegraph" in alnum:
        cues = (" my ", "show", "what", "progress", "contain", "topics", "snapshot")
        if not any(cue in f" {lowered} " for cue in cues):
            return False
        return True
    return False


def _is_review_request(text: str) -> bool:
    lowered = _norm_text(text)
    return any(keyword in lowered for keyword in REVIEW_KEYWORDS)


def _is_learning_intent(text: str) -> bool:
    lowered = _norm_text(text)
    return any(keyword in lowered for keyword in LEARN_INTENT_KEYWORDS)


def _is_teach_request(text: str) -> bool:
    lowered = _norm_text(text)
    return any(keyword in lowered for keyword in TEACH_INTENT_KEYWORDS)


def _is_social_message(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    alnum = _norm_alnum(text)
    if alnum in {"hey", "hi", "hello", "thanks", "thankyou"}:
        return True
    if len(lowered.split()) <= 7 and any(keyword in lowered for keyword in SOCIAL_KEYWORDS):
        return True
    return False


def _is_defer_review_request(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    return any(keyword in lowered for keyword in DEFER_REVIEW_KEYWORDS)


def _is_inability_response(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    patterns = (
        "i cant",
        "i can't",
        "cant",
        "can't",
        "i dont know",
        "i don't know",
        "dont know",
        "don't know",
        "idk",
        "not sure",
        "no idea",
    )
    return any(pattern in lowered for pattern in patterns)


def _is_resume_request(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    patterns = (
        "continue",
        "resume",
        "pick up where we left",
        "quick check",
        "pending question",
    )
    return any(pattern in lowered for pattern in patterns)


def _set_active_review_status(review: Dict[str, Any], status: str) -> Dict[str, Any]:
    row = dict(review or {})
    if status not in {"awaiting_answer", "paused", "completed"}:
        status = "completed"
    row["status"] = status
    row["awaiting_answer"] = status == "awaiting_answer"
    return row


def _active_review_awaiting(model: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    active_review = model.get("active_review")
    if not isinstance(active_review, dict):
        return {}, False
    status = str(active_review.get("status") or "").strip()
    if status in {"awaiting_answer", "paused", "completed"}:
        awaiting = status == "awaiting_answer"
    else:
        awaiting = bool(active_review.get("awaiting_answer"))
        status = "awaiting_answer" if awaiting else "completed"
    active_review = _set_active_review_status(active_review, status)
    return active_review, awaiting


def _pop_paused_review(model: Dict[str, Any]) -> Dict[str, Any] | None:
    paused = model.get("paused_reviews")
    if not isinstance(paused, list) or not paused:
        return None
    item = paused.pop()
    if not isinstance(item, dict):
        model["paused_reviews"] = paused
        return None
    model["paused_reviews"] = paused
    return _set_active_review_status(item, "awaiting_answer")


def _has_explicit_task_intent(text: str) -> bool:
    return (
        _is_progress_request(text)
        or _is_review_request(text)
        or _is_learning_intent(text)
        or _is_teach_request(text)
    )


def _topic_has_history(model: Dict[str, Any], topic: str) -> bool:
    topic_name = canonicalize_topic_label(topic or "")
    if not topic_name:
        return False
    safe = ensure_model_shape(model)
    topics = safe.get("topics")
    if not isinstance(topics, dict):
        return False
    topic_data = topics.get(topic_name)
    if not isinstance(topic_data, dict):
        return False
    concepts = topic_data.get("concepts")
    if not isinstance(concepts, dict):
        return False
    return bool(concepts)


def _is_assessment_request(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    if _is_review_request(text):
        return True
    patterns = (
        "quick check",
        "diagnostic",
        "assess",
        "am i ready",
        "test my",
        "quiz me",
    )
    return any(pattern in lowered for pattern in patterns)


def _is_depth_request(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    patterns = (
        "go deeper",
        "in depth",
        "in detail",
        "deep dive",
        "advanced",
        "technical",
        "derive",
        "derivation",
        "math",
    )
    return any(pattern in lowered for pattern in patterns)


def _extract_target_topic(text: str) -> str:
    lowered = _norm_text(text)
    if not lowered:
        return ""

    candidate = ""
    markers = (
        "refresh my memory on",
        "refresh myself on",
        "refresh me on",
        "learn about",
        "learn",
        "refresh",
        "review",
        "teach me about",
        "teach me",
        "explain",
        "help me learn",
        "what is",
        "what are",
    )
    for marker in markers:
        if marker in lowered:
            candidate = lowered.split(marker, 1)[1].strip()
            break

    if not candidate:
        return ""

    prefixes = (
        "my memory on ",
        "myself on ",
        "me on ",
        "about ",
        "on ",
        "the ",
        "my ",
        "myself ",
        "a ",
        "an ",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                changed = True

    candidate = _first_concept_fragment(candidate)
    if not candidate:
        return ""
    return canonicalize_topic_label(candidate.strip())


def _extract_progress_scope_topic(student_text: str, model: Dict[str, Any]) -> str:
    lowered = _norm_text(student_text)
    if not lowered:
        return ""
    if any(marker in lowered for marker in ("all topics", "overall", "across topics", "global progress")):
        return ""

    topics = model.get("topics")
    if not isinstance(topics, dict) or not topics:
        return ""

    candidates = sorted((str(name) for name in topics.keys()), key=len, reverse=True)
    for topic_name in candidates:
        topic_norm = _norm_text(topic_name)
        if not topic_norm:
            continue
        if topic_norm not in lowered:
            continue
        if any(f"{prefix}{topic_norm}" in lowered for prefix in ("for ", "on ", "in ", "about ")):
            return canonicalize_topic_label(topic_name)
    return ""


def _update_evidence_kind(update: Dict[str, Any]) -> str:
    evidence = update.get("evidence")
    if isinstance(evidence, dict):
        return str(evidence.get("kind") or "").strip().lower()
    return ""


def _topic_concepts(model: Dict[str, Any], topic: str) -> Dict[str, Any]:
    safe = ensure_model_shape(model)
    topics = safe.get("topics")
    if not isinstance(topics, dict):
        return {}
    topic_data = topics.get(canonicalize_topic_label(topic))
    if not isinstance(topic_data, dict):
        return {}
    concepts = topic_data.get("concepts")
    if not isinstance(concepts, dict):
        return {}
    return concepts


def _concept_base_key(text: str) -> str:
    raw = _norm_text(str(text or "").replace("_", " ").replace("-", " "))
    if not raw:
        return ""
    tokens = [token for token in raw.split() if token]
    if not tokens:
        return ""

    drop_tokens = {
        "concept",
        "concepts",
        "mechanism",
        "mechanisms",
        "basic",
        "basics",
        "overview",
        "intro",
        "introduction",
        "topic",
        "topics",
        "theory",
    }
    normalized: List[str] = []
    for token in tokens:
        cleaned = "".join(ch for ch in token if ch.isalnum())
        if not cleaned:
            continue
        if cleaned.endswith("s") and len(cleaned) > 3:
            cleaned = cleaned[:-1]
        if cleaned in drop_tokens:
            continue
        normalized.append(cleaned)
    if not normalized:
        normalized = ["".join(ch for ch in token if ch.isalnum()) for token in tokens]
        normalized = [token for token in normalized if token]
    return " ".join(normalized)


def _find_similar_concept_in_topic(
    model: Dict[str, Any], topic: str, concept_id: str, label: str
) -> tuple[str, str]:
    concepts = _topic_concepts(model, topic)
    if not concepts:
        return ("", "")

    wanted = _concept_base_key(label) or _concept_base_key(concept_id)
    if not wanted:
        return ("", "")

    for existing_id, meta in concepts.items():
        if not isinstance(existing_id, str) or not isinstance(meta, dict):
            continue
        existing_label = str(meta.get("label") or existing_id)
        existing_key = _concept_base_key(existing_label) or _concept_base_key(existing_id)
        if not existing_key:
            continue
        if existing_key == wanted:
            return (existing_id, existing_label)
    return ("", "")


def _concept_exists(model: Dict[str, Any], topic: str, concept_id: str, label: str) -> bool:
    concepts = _topic_concepts(model, topic)
    if not concepts:
        return False

    wanted_id = _to_optional_concept_id(concept_id)
    wanted_label = _norm_text(label)
    for existing_id, meta in concepts.items():
        if not isinstance(existing_id, str) or not isinstance(meta, dict):
            continue
        existing_label = _norm_text(str(meta.get("label") or existing_id))
        if wanted_id and _to_optional_concept_id(existing_id) == wanted_id:
            return True
        if wanted_label and existing_label == wanted_label:
            return True
    return False


def _is_meta_progress_text(text: str) -> bool:
    lowered = _norm_text(text)
    if not lowered:
        return False
    markers = (
        "show my progress",
        "knowledge graph",
        "student model",
        "what are my topics",
        "what do i know",
    )
    return any(marker in lowered for marker in markers)


def _filter_concept_updates_for_persistence(
    *,
    model: Dict[str, Any],
    updates: List[Dict[str, Any]],
    route_intent: str,
    default_topic: str,
) -> List[Dict[str, Any]]:
    if not isinstance(updates, list):
        return []

    intent = (route_intent or "").strip().lower()
    if intent in {"session_start", "progress", "pending_social", "pending_defer"}:
        return []

    by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for raw in updates:
        if not isinstance(raw, dict):
            continue

        topic = canonicalize_topic_label(str(raw.get("topic") or default_topic or "General"))
        label = str(raw.get("label") or "").strip()
        concept_id = _to_optional_concept_id(str(raw.get("concept_id") or label))
        if not concept_id:
            continue
        if not label:
            label = concept_id.replace("_", " ").title()

        row = dict(raw)
        row["topic"] = topic
        similar_id, similar_label = _find_similar_concept_in_topic(
            model,
            topic=topic,
            concept_id=concept_id,
            label=label,
        )
        if similar_id:
            concept_id = similar_id
            label = similar_label or label
        row["concept_id"] = concept_id
        row["label"] = label
        row["proficiency_delta"] = max(-0.2, min(0.2, _fallback_delta_for_update(row)))

        kind = _update_evidence_kind(row)
        evidence = row.get("evidence")
        evidence_text = ""
        if isinstance(evidence, dict):
            evidence_text = str(evidence.get("text") or "")
        is_meta = _is_meta_progress_text(evidence_text)

        exists = bool(similar_id) or _concept_exists(model, topic, concept_id, label)
        delta = abs(float(row.get("proficiency_delta", 0.0)))

        # Drop weak mention/observation writes for new concepts.
        if not exists and kind in {"mentioned", "observation"} and delta < 0.05:
            continue
        # Do not persist meta/progress phrasing as concept knowledge.
        if kind in {"mentioned", "observation"} and is_meta:
            continue
        # Chat intent should only persist stronger signals.
        if intent == "chat" and kind in {"mentioned", "observation", "asked"} and delta < 0.1:
            continue

        key = (topic, concept_id)
        prev = by_key.get(key)
        if not isinstance(prev, dict):
            by_key[key] = row
            continue

        prev_delta = abs(float(prev.get("proficiency_delta", 0.0)))
        if delta > prev_delta:
            by_key[key] = row

    return list(by_key.values())[:8]


def _is_specific_goal_concept(concept_id: str) -> bool:
    cid = _to_optional_concept_id(concept_id)
    if not cid:
        return False
    return cid not in {
        "concept",
        "topic",
        "subject",
        "general",
        "this_topic",
        "learning",
        "today",
        "tomorrow",
        "now",
        "anything",
        "something",
        "stuff",
        "question",
        "review",
    }


def _is_generic_gate_candidate(planned: Dict[str, Any]) -> bool:
    label = _norm_text(str(planned.get("label") or ""))
    question = _norm_text(str(planned.get("question") or ""))
    concept_id = _to_optional_concept_id(str(planned.get("concept_id") or ""))
    if concept_id in {"concept", "topic", "subject"}:
        return True
    if label in {"concept", "topic", "subject", "foundation concept"}:
        return True
    if "what topic are you interested" in question:
        return True
    return False


def _topic_exists(model: Dict[str, Any], topic: str) -> bool:
    topic_name = canonicalize_topic_label(topic or "")
    if not topic_name:
        return False
    safe = ensure_model_shape(model)
    topics = safe.get("topics")
    return isinstance(topics, dict) and topic_name in topics


def _pedagogical_hint_from_plan(plan: Dict[str, Any]) -> str:
    if not isinstance(plan, dict):
        return ""
    mode = str(plan.get("plan_mode") or "").strip()
    target_concept = str(plan.get("target_concept") or "").strip()
    misconception = str(plan.get("misconception") or "").strip()
    instructions = str(plan.get("response_instructions") or "").strip()

    pieces: List[str] = []
    if mode == "explain_then_probe":
        pieces.append("Answer the exact confusion first, then ask one short comprehension check.")
    elif mode == "probe_only":
        pieces.append("Ask one concise comprehension probe before adding more explanation.")
    elif mode == "teach_gap_then_resume":
        pieces.append("Teach the identified gap briefly, then reconnect to the original question.")
    elif mode == "social_reanchor":
        pieces.append("Reply socially in one sentence, then re-anchor to the study goal.")
    else:
        pieces.append("Give a direct, plain-language answer tailored to the student question.")

    if target_concept:
        pieces.append(f"Target concept: {target_concept}.")
    if misconception:
        pieces.append(f"Misconception to correct: {misconception}.")
    if instructions:
        pieces.append(f"Extra guidance: {instructions}")
    return " ".join(pieces).strip()


def _pedagogical_plan_with_reasoning(
    *,
    student_text: str,
    route: Dict[str, Any],
    model: Dict[str, Any],
    current_topic: str,
) -> Dict[str, Any]:
    route_intent = str(route.get("intent") or "chat")
    prompt = PEDAGOGICAL_PLAN_PROMPT.format(
        student_message=student_text or "(empty)",
        route_intent=route_intent,
        current_topic=current_topic or "General",
        progress_snapshot=model_snapshot(model, current_topic=current_topic or "General"),
        review_snapshot=review_snapshot(model, current_topic=current_topic, limit=3),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {}

    mode = str(data.get("plan_mode") or "").strip().lower()
    valid_modes = {"direct_answer", "explain_then_probe", "probe_only", "teach_gap_then_resume", "social_reanchor"}
    if mode not in valid_modes:
        mode = ""

    target_topic = str(data.get("target_topic") or "").strip()
    if not target_topic:
        target_topic = str(route.get("target_topic") or "").strip() or current_topic or "General"
    target_topic = canonicalize_topic_label(target_topic)

    target_concept = str(data.get("target_concept") or "").strip()
    misconception = str(data.get("misconception") or "").strip()
    response_instructions = str(data.get("response_instructions") or "").strip()

    raw_gaps = data.get("suspected_gaps")
    suspected_gaps: List[Dict[str, Any]] = []
    if isinstance(raw_gaps, list):
        for gap in raw_gaps:
            if not isinstance(gap, dict):
                continue
            label = str(gap.get("label") or "").strip()
            concept_id = _to_optional_concept_id(str(gap.get("concept_id") or label))
            if not label and not concept_id:
                continue
            suspected_gaps.append(
                {
                    "concept_id": concept_id,
                    "label": label or concept_id.replace("_", " ").title(),
                    "reason": str(gap.get("reason") or "").strip(),
                    "confidence": _clamp01(gap.get("confidence", 0.0)),
                }
            )
            if len(suspected_gaps) >= 3:
                break

    if not mode:
        if route_intent == "pending_social":
            mode = "social_reanchor"
        elif _is_inability_response(student_text):
            mode = "teach_gap_then_resume"
        elif "confused" in _norm_text(student_text) or "not sure" in _norm_text(student_text):
            mode = "explain_then_probe"
        elif route_intent in {"learn_request", "pending_switch"}:
            mode = "explain_then_probe"
        else:
            mode = "direct_answer"

    if not response_instructions:
        response_instructions = "Keep the response focused and conversational."

    return {
        "plan_mode": mode,
        "target_topic": target_topic,
        "target_concept": target_concept,
        "misconception": misconception,
        "response_instructions": response_instructions,
        "suspected_gaps": suspected_gaps,
    }


def _route_intent_with_reasoning(
    *,
    student_text: str,
    current_topic: str,
    prereq_context: bool,
    model: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = INTENT_ROUTER_PROMPT.format(
        student_message=student_text or "(empty)",
        current_topic=current_topic or "General",
        prereq_context="yes" if prereq_context else "no",
        review_snapshot=review_snapshot(model, current_topic=current_topic, limit=3),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {}

    next_action = str(data.get("next_action") or "").strip().lower()
    valid = {"chat", "progress_report", "ask_review_question", "teach_foundation", "plan_next_step"}
    if next_action not in valid:
        next_action = ""
    if next_action == "teach_foundation" and not prereq_context:
        next_action = "plan_next_step"

    reason = str(data.get("reason") or "").strip()
    confidence = _clamp01(data.get("confidence", 0.0))

    if not next_action:
        if _is_progress_request(student_text):
            next_action = "progress_report"
            reason = reason or "fallback keyword progress intent"
            confidence = max(confidence, 0.78)
        elif _is_review_request(student_text):
            next_action = "ask_review_question"
            reason = reason or "fallback keyword review intent"
            confidence = max(confidence, 0.74)
        elif prereq_context and _is_teach_request(student_text):
            next_action = "teach_foundation"
            reason = reason or "fallback keyword prereq-teach intent"
            confidence = max(confidence, 0.7)
        elif _is_social_message(student_text):
            next_action = "chat"
            reason = reason or "fallback social-chat intent"
            confidence = max(confidence, 0.68)
        else:
            next_action = "plan_next_step"
            reason = reason or "fallback to planner"
            confidence = max(confidence, 0.4)
    elif not reason:
        reason = "semantic intent routing"
    elif confidence <= 0.0:
        confidence = 0.6

    return {"next_action": next_action, "reason": reason, "confidence": confidence}


def _pause_active_review(*, model: Dict[str, Any], student_text: str, reason: str) -> Dict[str, Any]:
    safe = ensure_model_shape(model)
    active_review, awaiting_answer = _active_review_awaiting(safe)
    if not awaiting_answer:
        return safe

    paused_reviews = safe.get("paused_reviews")
    if not isinstance(paused_reviews, list):
        paused_reviews = []

    paused = _set_active_review_status(active_review, "paused")
    paused["paused_at"] = _utc_now_iso()
    paused["pause_reason"] = reason
    if student_text:
        paused["paused_by"] = student_text[:180]
    paused_reviews.append(paused)

    safe["paused_reviews"] = paused_reviews[-8:]
    safe["active_review"] = {}
    return safe


def _route_pending_review_with_reasoning(
    *,
    student_text: str,
    model: Dict[str, Any],
    active_review: Dict[str, Any],
    current_topic: str,
    prereq_context: bool,
) -> Dict[str, Any]:
    question = str(active_review.get("question") or "").strip()
    if not question:
        question = "Please answer the pending review question in your own words."
    concept = str(active_review.get("label") or active_review.get("concept_id") or "current concept")

    prompt = PENDING_REVIEW_ROUTER_PROMPT.format(
        student_message=student_text or "(empty)",
        pending_question=question,
        pending_concept=concept,
        current_topic=current_topic or "General",
        prereq_context="yes" if prereq_context else "no",
        review_snapshot=review_snapshot(model, current_topic=current_topic, limit=3),
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {}

    next_action = str(data.get("next_action") or "").strip().lower()
    valid = {
        "grade_review_answer",
        "teach_foundation",
        "remind_review_question",
        "progress_report",
        "chat",
        "plan_next_step",
    }
    if next_action not in valid:
        next_action = ""

    reason = str(data.get("reason") or "").strip()
    confidence = _clamp01(data.get("confidence", 0.0))
    pause_pending_review = bool(data.get("pause_pending_review"))
    if not next_action:
        if not student_text:
            next_action = "remind_review_question"
            reason = reason or "empty response while review pending"
            confidence = max(confidence, 0.78)
        elif _is_defer_review_request(student_text):
            next_action = "remind_review_question"
            reason = reason or "fallback defer pending review"
            confidence = max(confidence, 0.72)
        elif _is_progress_request(student_text):
            next_action = "progress_report"
            reason = reason or "fallback progress request"
            confidence = max(confidence, 0.78)
        elif _is_review_request(student_text):
            next_action = "remind_review_question"
            reason = reason or "fallback review request while pending"
            confidence = max(confidence, 0.68)
        elif _is_learning_intent(student_text) or _is_teach_request(student_text):
            next_action = "plan_next_step"
            reason = reason or "fallback task switch while review pending"
            confidence = max(confidence, 0.62)
            pause_pending_review = True
        elif _is_social_message(student_text) and not _has_explicit_task_intent(student_text):
            next_action = "chat"
            reason = reason or "fallback social message"
            confidence = max(confidence, 0.66)
        else:
            next_action = "grade_review_answer"
            reason = reason or "fallback treat message as review attempt"
            confidence = max(confidence, 0.42)
    elif not reason:
        reason = "semantic pending-review routing"
    elif confidence <= 0.0:
        confidence = 0.6

    if next_action == "plan_next_step":
        pause_pending_review = True
    else:
        pause_pending_review = False

    return {
        "next_action": next_action,
        "reason": reason,
        "confidence": confidence,
        "pause_pending_review": pause_pending_review,
    }


def _route_turn_with_reasoning(
    *,
    student_text: str,
    model: Dict[str, Any],
    current_topic: str,
    active_review: Dict[str, Any],
    awaiting_answer: bool,
) -> Dict[str, Any]:
    pending_question = ""
    pending_concept = ""
    if awaiting_answer:
        pending_question = str(active_review.get("question") or "").strip()
        pending_concept = str(active_review.get("label") or active_review.get("concept_id") or "current concept")

    paused_reviews = model.get("paused_reviews")
    paused_count = len(paused_reviews) if isinstance(paused_reviews, list) else 0

    prompt = TURN_ROUTER_PROMPT.format(
        student_message=student_text or "(empty)",
        current_topic=current_topic or "General",
        has_pending_review="yes" if awaiting_answer else "no",
        pending_question=pending_question or "(none)",
        pending_concept=pending_concept or "(none)",
        paused_count=paused_count,
        review_snapshot=review_snapshot(model, current_topic=current_topic, limit=4),
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {}

    valid_intents = {
        "session_start",
        "pending_social",
        "pending_answer",
        "pending_defer",
        "pending_switch",
        "progress",
        "review_request",
        "learn_request",
        "chat",
    }
    intent = str(data.get("intent") or "").strip().lower()
    if intent not in valid_intents:
        intent = ""

    reason = str(data.get("reason") or "").strip()
    confidence = _clamp01(data.get("confidence", 0.0))
    pause_pending = bool(data.get("pause_pending"))
    resume_pending = bool(data.get("resume_pending"))
    target_topic = str(data.get("target_topic") or "").strip()
    inferred_topic = _extract_target_topic(student_text)

    if intent:
        pending_allowed = {
            "session_start",
            "pending_social",
            "pending_answer",
            "pending_defer",
            "pending_switch",
            "progress",
        }
        non_pending_allowed = {
            "session_start",
            "progress",
            "review_request",
            "learn_request",
            "chat",
        }
        if awaiting_answer and intent not in pending_allowed:
            intent = ""
        elif not awaiting_answer and intent not in non_pending_allowed:
            intent = ""

    if not intent:
        if not student_text:
            intent = "session_start"
            reason = reason or "empty user message"
            confidence = max(confidence, 0.8)
        elif awaiting_answer:
            if _is_progress_request(student_text):
                intent = "progress"
                reason = reason or "fallback progress request while pending"
                confidence = max(confidence, 0.78)
            elif _is_defer_review_request(student_text):
                intent = "pending_defer"
                reason = reason or "fallback defer while pending"
                confidence = max(confidence, 0.74)
            elif _is_social_message(student_text) and not _has_explicit_task_intent(student_text):
                intent = "pending_social"
                reason = reason or "fallback social while pending"
                confidence = max(confidence, 0.7)
            elif _is_learning_intent(student_text) or _is_teach_request(student_text):
                intent = "pending_switch"
                reason = reason or "fallback topic/task switch while pending"
                pause_pending = True
                if not target_topic:
                    target_topic = inferred_topic
                confidence = max(confidence, 0.68)
            else:
                intent = "pending_answer"
                reason = reason or "fallback treat as pending answer"
                confidence = max(confidence, 0.55)
        else:
            if _is_progress_request(student_text):
                intent = "progress"
                reason = reason or "fallback progress request"
                confidence = max(confidence, 0.78)
            elif _is_review_request(student_text):
                intent = "review_request"
                reason = reason or "fallback review request"
                confidence = max(confidence, 0.72)
            elif _is_learning_intent(student_text) or _is_teach_request(student_text):
                intent = "learn_request"
                reason = reason or "fallback learn request"
                if not target_topic:
                    target_topic = inferred_topic
                confidence = max(confidence, 0.68)
            elif _is_social_message(student_text):
                intent = "chat"
                reason = reason or "fallback social chat"
                confidence = max(confidence, 0.64)
            else:
                intent = "chat"
                reason = reason or "fallback chat"
                confidence = max(confidence, 0.5)
    elif not reason:
        reason = "semantic turn routing"
        if confidence <= 0.0:
            confidence = 0.6

    if awaiting_answer and intent == "pending_switch":
        pause_pending = True
        if not target_topic:
            target_topic = inferred_topic
    if intent != "pending_switch":
        pause_pending = False
    if intent == "session_start":
        resume_pending = False

    if not awaiting_answer and _is_resume_request(student_text) and paused_count > 0:
        intent = "review_request"
        resume_pending = True
        reason = "resume request for paused quick check"
        confidence = max(confidence, 0.75)
    if intent in {"learn_request", "pending_switch"} and not target_topic:
        target_topic = inferred_topic
    if target_topic:
        target_topic = canonicalize_topic_label(target_topic)

    return {
        "intent": intent,
        "reason": reason,
        "confidence": confidence,
        "pause_pending": pause_pending,
        "resume_pending": resume_pending,
        "target_topic": target_topic,
    }


def route_turn(state: TutorState) -> Dict[str, Any]:
    student_text = (state.get("student_answer") or "").strip()
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    active_review, awaiting_answer = _active_review_awaiting(model)
    if active_review:
        model["active_review"] = active_review

    route = _route_turn_with_reasoning(
        student_text=student_text,
        model=model,
        current_topic=current_topic,
        active_review=active_review,
        awaiting_answer=awaiting_answer,
    )
    return {
        "student_model": model,
        "route": route,
        "response_mode": str(route.get("intent") or ""),
        "action_reason": str(route.get("reason") or ""),
    }


def pedagogical_plan(state: TutorState) -> Dict[str, Any]:
    student_text = (state.get("student_answer") or "").strip()
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    route = state.get("route")
    if not isinstance(route, dict):
        route = {}

    if not student_text:
        return {"pedagogical_plan": {}}

    plan = _pedagogical_plan_with_reasoning(
        student_text=student_text,
        route=route,
        model=model,
        current_topic=current_topic,
    )
    return {"pedagogical_plan": plan}


def _pending_context_message(active_review: Dict[str, Any]) -> str:
    question = str(active_review.get("question") or "").strip()
    if not question:
        question = "Please answer the pending quick check in your own words."
    goal = str(active_review.get("goal_concept") or "").replace("_", " ").strip()
    if goal:
        header = f"You have an unfinished quick check before we continue with {goal}:"
    else:
        header = "You have an unfinished quick check:"
    return (
        f"{header}\n{question}\n\n"
        "You can answer now, ask me to teach it first, or switch topics and I will pause this check."
    )


def _run_learning_flow(state: TutorState, model: Dict[str, Any]) -> Dict[str, Any]:
    plan = state.get("pedagogical_plan")
    if not isinstance(plan, dict):
        plan = {}
    plan_mode = str(plan.get("plan_mode") or "").strip()
    plan_hint = _pedagogical_hint_from_plan(plan)

    if plan_mode in {"direct_answer", "explain_then_probe", "probe_only", "teach_gap_then_resume"}:
        merged_hint = " ".join(
            part for part in [(state.get("tutor_hint") or "").strip(), plan_hint] if part
        ).strip()
        return chat({**state, "student_model": model, "tutor_hint": merged_hint})

    planning = plan_next_step({**state, "student_model": model})
    planned_model = ensure_model_shape(planning.get("student_model") or model)
    next_action = str(planning.get("next") or "chat")

    if next_action == "ask_prereq_diagnostic":
        out = ask_prereq_diagnostic({**state, **planning, "student_model": planned_model})
    elif next_action == "ask_review_question":
        out = ask_review_question({**state, **planning, "student_model": planned_model})
    else:
        merged_hint = " ".join(
            part for part in [(state.get("tutor_hint") or "").strip(), plan_hint] if part
        ).strip()
        out = chat({**state, **planning, "student_model": planned_model, "tutor_hint": merged_hint})
    return out


def respond_turn(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    route = state.get("route")
    if not isinstance(route, dict):
        route = {}
    pedagogical = state.get("pedagogical_plan")
    if not isinstance(pedagogical, dict):
        pedagogical = {}
    intent = str(route.get("intent") or "chat")
    pause_pending = bool(route.get("pause_pending"))
    resume_pending = bool(route.get("resume_pending"))
    target_topic = str(route.get("target_topic") or "").strip()
    plan_hint = _pedagogical_hint_from_plan(pedagogical)

    if resume_pending:
        active_review, awaiting_answer = _active_review_awaiting(model)
        if not awaiting_answer:
            resumed = _pop_paused_review(model)
            if resumed:
                model["active_review"] = resumed

    active_review, awaiting_answer = _active_review_awaiting(model)
    if active_review:
        model["active_review"] = active_review

    if pause_pending and awaiting_answer:
        model = _pause_active_review(
            model=model,
            student_text=student_text,
            reason=str(route.get("reason") or "student switched tasks during pending review"),
        )
        active_review, awaiting_answer = _active_review_awaiting(model)

    if intent == "session_start":
        if awaiting_answer and isinstance(active_review, dict):
            message = _pending_context_message(active_review)
        else:
            paused_reviews = model.get("paused_reviews")
            paused_count = len(paused_reviews) if isinstance(paused_reviews, list) else 0
            if paused_count > 0:
                message = (
                    "I can help you learn continuously and keep your progress.\n\n"
                    f"You also have {paused_count} paused quick check(s). "
                    "Say \"resume quick check\" anytime to pick one up."
                )
            else:
                message = (
                    "I can help you learn continuously and keep your progress.\n\n"
                    "Try one of these:\n"
                    "- Ask a topic question\n"
                    "- Say \"what should I review?\"\n"
                    "- Say \"show my progress\""
                )
        return {
            "messages": [AIMessage(content=message)],
            "consumed_student_text": "",
            "student_answer": "",
            "student_model": model,
            "response_mode": intent,
            "pending_context_message": message if awaiting_answer else "",
        }

    if intent == "pending_social" and awaiting_answer and isinstance(active_review, dict):
        question = str(active_review.get("question") or "").strip()
        if not question:
            question = "Please answer the pending quick check in your own words."
        social = "I'm doing well and ready to help."
        message = (
            f"{social} Do you want to continue our quick check?\n\n"
            f"Pending question:\n{question}"
        )
        return {
            "messages": _append_turn_messages(student_text, message),
            "consumed_student_text": student_text,
            "student_answer": "",
            "student_model": model,
            "response_mode": intent,
            "pending_context_message": question,
        }

    if intent == "pending_defer" and awaiting_answer and isinstance(active_review, dict):
        question = str(active_review.get("question") or "").strip()
        if not question:
            question = "Please answer the pending quick check in your own words."
        if pause_pending:
            model = _pause_active_review(
                model=model,
                student_text=student_text,
                reason=str(route.get("reason") or "student deferred pending review"),
            )
            message = (
                "No problem. I paused this quick check.\n"
                "Say \"resume quick check\" whenever you want to continue."
            )
        else:
            message = (
                "No problem, we can pause this for now.\n"
                "When you're ready, answer this quick check:\n"
                f"{question}"
            )
        return {
            "messages": _append_turn_messages(student_text, message),
            "consumed_student_text": student_text,
            "student_answer": "",
            "student_model": model,
            "response_mode": intent,
        }

    if intent == "progress":
        out = progress_report({**state, "student_model": model})
        out["student_model"] = model
        out["response_mode"] = intent
        return out

    if intent == "pending_switch":
        if awaiting_answer:
            model = _pause_active_review(
                model=model,
                student_text=student_text,
                reason=str(route.get("reason") or "student switched topic"),
            )
        current_topic = canonicalize_topic_label(state.get("current_topic") or "General")
        if target_topic:
            current_topic = canonicalize_topic_label(target_topic)
        out = _run_learning_flow({**state, "current_topic": current_topic, "student_model": model}, model)
        out["student_model"] = ensure_model_shape(out.get("student_model") or model)
        out["response_mode"] = intent
        return out

    if intent == "review_request":
        out = ask_review_question({**state, "student_model": model})
        out["student_model"] = ensure_model_shape(out.get("student_model") or model)
        out["response_mode"] = intent
        return out

    if intent == "learn_request":
        current_topic = canonicalize_topic_label(state.get("current_topic") or "General")
        if target_topic:
            current_topic = canonicalize_topic_label(target_topic)
        out = _run_learning_flow({**state, "current_topic": current_topic, "student_model": model}, model)
        out["student_model"] = ensure_model_shape(out.get("student_model") or model)
        out["response_mode"] = intent
        return out

    if intent == "pending_answer" and awaiting_answer:
        if _is_inability_response(student_text):
            out = teach_foundation({**state, "student_model": model})
            out["student_model"] = ensure_model_shape(out.get("student_model") or model)
            out["response_mode"] = "teach_then_reask"
            return out
        out = grade_review_answer({**state, "student_model": model})
        out["student_model"] = ensure_model_shape(out.get("student_model") or model)
        out["response_mode"] = intent
        return out

    merged_hint = " ".join(part for part in [(state.get("tutor_hint") or "").strip(), plan_hint] if part).strip()
    out = chat({**state, "student_model": model, "tutor_hint": merged_hint})
    out["student_model"] = ensure_model_shape(out.get("student_model") or model)
    out["response_mode"] = "chat"
    return out


def load_student_model(state: TutorState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id") or "default"
    model = store_load_student_model(student_id=str(thread_id))
    return {
        "student_model": ensure_model_shape(model),
        "concept_updates": [],
        "review_result": {},
        "planned_review_target": {},
        "planned_goal_concept": "",
        "route": {},
        "pedagogical_plan": {},
        "understanding_assessment": {},
        "response_mode": "",
        "pending_context_message": "",
        "action_reason": "",
    }


def router(state: TutorState) -> Dict[str, Any]:
    student_text = (state.get("student_answer") or "").strip()
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    active_review = model.get("active_review", {})
    awaiting_answer = bool(active_review.get("awaiting_answer")) if isinstance(active_review, dict) else False
    prereq_mode = str(active_review.get("mode") or "") if isinstance(active_review, dict) else ""
    prereq_context = (
        prereq_mode == "prereq_gate"
        and bool(str(active_review.get("concept_id") or "").strip())
        and bool(str(active_review.get("goal_concept") or "").strip())
    ) if isinstance(active_review, dict) else False

    if awaiting_answer:
        decision = _route_pending_review_with_reasoning(
            student_text=student_text,
            model=model,
            active_review=active_review if isinstance(active_review, dict) else {},
            current_topic=current_topic,
            prereq_context=prereq_context,
        )
        next_action = str(decision.get("next_action") or "remind_review_question")
        reason = str(decision.get("reason") or "semantic pending-review routing")
        pending_confidence = _clamp01(decision.get("confidence", 0.0))
        pause_pending_review = bool(decision.get("pause_pending_review"))

        # Let a strong general-intent signal break out of pending-review mode.
        if student_text and next_action in {"grade_review_answer", "teach_foundation", "remind_review_question"}:
            general_decision = _route_intent_with_reasoning(
                student_text=student_text,
                current_topic=current_topic,
                prereq_context=prereq_context,
                model=model,
            )
            general_action = str(general_decision.get("next_action") or "")
            general_confidence = _clamp01(general_decision.get("confidence", 0.0))
            if general_action in {"plan_next_step", "progress_report"} and general_confidence >= max(0.65, pending_confidence + 0.15):
                next_action = general_action
                reason = f"intent override from general router: {general_decision.get('reason', 'semantic intent')}"
                pause_pending_review = general_action == "plan_next_step"

        result: Dict[str, Any] = {
            "next": next_action,
            "action_reason": reason,
        }
        if pause_pending_review:
            result["student_model"] = _pause_active_review(model=model, student_text=student_text, reason=reason)

        return result

    if not student_text:
        return {"next": "greet", "action_reason": "session kickoff"}

    if _is_social_message(student_text) and not _has_explicit_task_intent(student_text):
        return {"next": "chat", "action_reason": "social chat message"}

    decision = _route_intent_with_reasoning(
        student_text=student_text,
        current_topic=current_topic,
        prereq_context=prereq_context,
        model=model,
    )
    return {
        "next": decision.get("next_action", "plan_next_step"),
        "action_reason": decision.get("reason", "semantic intent routing"),
    }


def plan_next_step(state: TutorState) -> Dict[str, Any]:
    student_text = (state.get("student_answer") or "").strip()
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    progress = model_snapshot(model, current_topic=current_topic or "General")
    review = review_snapshot(model, current_topic=current_topic, limit=4)
    topic_has_history = _topic_has_history(model, current_topic)
    topic_history = "has_concept_history" if topic_has_history else "fresh_or_sparse_topic_history"

    planner_prompt = PREREQ_PLANNER_PROMPT.format(
        student_message=student_text or "(empty)",
        current_topic=current_topic or "General",
        progress_snapshot=progress,
        review_snapshot=review,
        topic_history=topic_history,
    )

    planner_data: Dict[str, Any] = {}
    try:
        planner_response = llm.invoke([HumanMessage(content=planner_prompt)])
        planner_data = extract_json(planner_response.content)
    except Exception:
        planner_data = {}

    next_action = str(planner_data.get("next_action") or "chat").strip().lower()
    if next_action not in {"chat", "ask_prereq_diagnostic", "ask_review_question"}:
        next_action = "chat"

    reason = str(planner_data.get("action_reason") or "planner defaulted").strip() or "planner defaulted"
    goal_concept = _to_optional_concept_id(str(planner_data.get("goal_concept") or "").strip())
    interrupt_for_review = bool(planner_data.get("interrupt_for_review"))
    should_gate = bool(planner_data.get("should_gate"))

    planned: Dict[str, Any] = {}
    raw_candidate = planner_data.get("gate_candidate")
    if isinstance(raw_candidate, dict):
        planned = _sanitize_gate_candidate(
            raw_candidate,
            current_topic=current_topic,
            goal_concept=goal_concept,
        )

    has_goal = _is_specific_goal_concept(goal_concept)
    wants_learning = _is_learning_intent(student_text) or next_action == "ask_prereq_diagnostic"
    explicit_assessment = _is_assessment_request(student_text)
    explicit_depth = _is_depth_request(student_text)
    fresh_topic_learn = (
        wants_learning and not topic_has_history and not explicit_assessment and not explicit_depth
    )

    if fresh_topic_learn:
        return {
            "next": "chat",
            "planned_goal_concept": goal_concept,
            "action_reason": "fresh-topic learning request: explain first, gate later if needed",
        }

    if (
        should_gate
        and planned
        and has_goal
        and wants_learning
        and not _is_generic_gate_candidate(planned)
        and (explicit_assessment or explicit_depth)
    ):
        confidence = _clamp01(planned.get("confidence", 0.0))
        estimated = _clamp01(planned.get("estimated_proficiency", 0.0))
        actual = _best_proficiency_for_concept(
            model,
            concept_id=str(planned.get("concept_id") or ""),
            label=str(planned.get("label") or ""),
        )
        weak_foundation = min(actual, estimated if estimated > 0 else actual) < 0.55
        likely_gap = weak_foundation and (confidence >= 0.58 or actual < 0.2)
        if likely_gap:
            return {
                "next": "ask_prereq_diagnostic",
                "planned_review_target": planned,
                "planned_goal_concept": goal_concept,
                "current_topic": canonicalize_topic_label(current_topic or str(planned.get("topic") or "General")),
                "action_reason": reason or "planner found likely prerequisite gap",
            }

    due = get_due_reviews(model, current_topic=current_topic, limit=1)
    turn_count = model.get("meta", {}).get("turn_count", 0) if isinstance(model.get("meta"), dict) else 0
    if due and (next_action == "ask_review_question" or interrupt_for_review):
        return {"next": "ask_review_question", "action_reason": reason or "planner suggested review"}

    if due and isinstance(turn_count, int) and turn_count > 0 and turn_count % 8 == 0:
        return {"next": "ask_review_question", "action_reason": "scheduled proactive review"}

    return {
        "next": "chat",
        "planned_goal_concept": goal_concept,
        "action_reason": reason or "normal tutoring response",
    }


def greet(state: TutorState) -> Dict[str, Any]:
    msg = (
        "I can help you learn continuously and keep your progress.\n\n"
        "Try one of these:\n"
        "- Ask a topic question\n"
        "- Say \"what should I review?\"\n"
        "- Say \"show my progress\""
    )
    return {
        "messages": [AIMessage(content=msg)],
        "consumed_student_text": "",
        "student_answer": "",
    }


def progress_report(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    scoped_topic = _extract_progress_scope_topic(student_text, model)

    topic_part = model_snapshot(model, current_topic=scoped_topic)
    review_part = review_snapshot(model, current_topic=scoped_topic, limit=5)
    msg = f"{topic_part}\n\n{review_part}"

    return {
        "messages": _append_turn_messages(student_text, msg),
        "consumed_student_text": student_text,
        "student_answer": "",
    }


def ask_prereq_diagnostic(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    planned = state.get("planned_review_target") or {}

    concept_id = str(planned.get("concept_id") or "concept").strip() or "concept"
    label = str(planned.get("label") or concept_id).strip() or concept_id
    goal_concept = str(planned.get("goal_concept") or state.get("planned_goal_concept") or "").strip()
    topic = canonicalize_topic_label(str(planned.get("topic") or state.get("current_topic") or "General"))
    question = str(planned.get("question") or "").strip()
    if not question:
        question = f"Before we dive deeper, quick check: can you explain {label} in your own words?"

    prompt = (
        f"Before we go deeper into {goal_concept.replace('_', ' ') if goal_concept else 'this topic'}, "
        f"I want to verify one foundation concept.\n\nQuick check:\n{question}"
    )

    model["active_review"] = {
        "status": "awaiting_answer",
        "awaiting_answer": True,
        "topic": topic,
        "concept_id": concept_id,
        "label": label,
        "question": question,
        "asked_at": _utc_now_iso(),
        "mode": "prereq_gate",
        "goal_concept": goal_concept,
    }

    concept_updates: List[Dict[str, Any]] = [
        {
            "topic": topic,
            "concept_id": concept_id,
            "label": label,
            "proficiency_delta": 0.02,
            "evidence": {
                "kind": "asked",
                "text": f"Prerequisite diagnostic opened for {label}",
                "source": "tutor",
                "reasoning": "foundation check should appear in concept graph",
            },
        }
    ]
    goal_id = _to_concept_id(goal_concept) if goal_concept else ""
    if goal_id and goal_id != concept_id:
        concept_updates.append(
            {
                "topic": topic,
                "concept_id": goal_id,
                "label": goal_concept.replace("_", " ").title(),
                "proficiency_delta": 0.02,
                "evidence": {
                    "kind": "asked",
                    "text": f"Student requested learning path toward {goal_concept.replace('_', ' ')}",
                    "source": "student",
                    "reasoning": "target concept should be captured when goal is stated",
                },
            }
        )

    return {
        "messages": _append_turn_messages(student_text, prompt),
        "consumed_student_text": student_text,
        "student_answer": "",
        "student_model": model,
        "concept_updates": concept_updates,
    }


def teach_foundation(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    active_review = model.get("active_review")
    if not isinstance(active_review, dict):
        return {
            "messages": _append_turn_messages(
                student_text,
                "Tell me which concept you want to build first, and I'll teach it step-by-step.",
            ),
            "consumed_student_text": student_text,
            "student_answer": "",
            "concept_updates": [],
        }

    label = str(active_review.get("label") or active_review.get("concept_id") or "this concept")
    goal_concept = str(active_review.get("goal_concept") or "your target topic").replace("_", " ")
    foundation_topic = canonicalize_topic_label(str(active_review.get("topic") or state.get("current_topic") or "General"))
    prompt = FOUNDATION_TEACH_PROMPT.format(
        concept_label=label,
        goal_concept=goal_concept,
        current_topic=foundation_topic,
        student_message=student_text or "(no extra request)",
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        explanation = (response.content or "").strip()
    except Exception:
        explanation = ""

    if not explanation:
        explanation = (
            f"Let's build {label} first.\n\n"
            f"A simple view: {label} is a mathematical building block used to represent information and transform it in models.\n"
            "In neural networks, we use these structures to pass data through layers and compute relationships.\n\n"
            f"Quick check: can you explain {label} and why it matters for {goal_concept}?"
        )
    fallback_question = f"Can you explain {label} and why it matters for {goal_concept}?"
    quick_check_question = _extract_quick_check_question(explanation) or fallback_question
    if quick_check_question not in explanation:
        explanation = f"{explanation}\n\nQuick check:\n{quick_check_question}"

    active_review = _set_active_review_status(active_review, "awaiting_answer")
    active_review["question"] = quick_check_question
    active_review["asked_at"] = _utc_now_iso()
    model["active_review"] = active_review

    concept_id = str(active_review.get("concept_id") or _to_concept_id(label))
    concept_updates = [
        {
            "topic": foundation_topic,
            "concept_id": concept_id,
            "label": label,
            "proficiency_delta": 0.04,
            "evidence": {
                "kind": "introduced",
                "text": f"Tutor taught foundation concept: {label}",
                "source": "tutor",
                "reasoning": "student requested foundational teaching",
            },
        }
    ]
    goal_id = _to_concept_id(str(active_review.get("goal_concept") or ""))
    if goal_id and goal_id != concept_id:
        concept_updates.append(
            {
                "topic": foundation_topic,
                "concept_id": goal_id,
                "label": str(active_review.get("goal_concept") or goal_id).replace("_", " ").title(),
                "proficiency_delta": 0.02,
                "evidence": {
                    "kind": "introduced",
                    "text": f"Tutor linked {label} to goal concept {goal_id.replace('_', ' ')}",
                    "source": "tutor",
                    "reasoning": "goal concept stays visible during prerequisite teaching",
                },
            }
        )

    return {
        "messages": _append_turn_messages(student_text, explanation),
        "consumed_student_text": student_text,
        "student_answer": "",
        "student_model": model,
        "concept_updates": concept_updates,
    }


def remind_review_question(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    active_review, awaiting_answer = _active_review_awaiting(model)
    if active_review:
        model["active_review"] = active_review

    if not awaiting_answer:
        message = "There is no pending review question right now."
    else:
        question = str(active_review.get("question") or "").strip()
        if not question:
            question = "Please answer the pending review question in your own words."
        mode = str(active_review.get("mode") or "")
        goal = str(active_review.get("goal_concept") or "").replace("_", " ").strip()
        if _is_defer_review_request(student_text):
            if mode == "prereq_gate" and goal:
                message = (
                    f"No problem, we can pause this quick check and keep going with {goal}.\n"
                    "When you're ready, answer this question:\n"
                    f"{question}"
                )
            else:
                message = (
                    "No problem, we can pause this review and continue.\n"
                    "When you're ready, answer this question:\n"
                    f"{question}"
                )
        elif mode == "prereq_gate" and goal:
            message = (
                f"You have an unfinished quick check before we continue with {goal}:\n"
                f"{question}\n\n"
                "You can answer now, ask me to teach it first, or switch topics and I will pause this check."
            )
        else:
            message = (
                f"You still have a pending review question:\n{question}\n\n"
                "You can answer now, or switch topics and I will pause this check."
            )

    return {
        "messages": _append_turn_messages(student_text, message),
        "consumed_student_text": student_text,
        "student_answer": "",
        "student_model": model,
    }


def ask_review_question(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    student_text = (state.get("student_answer") or "").strip()
    explicit_review_request = _is_review_request(student_text)

    active_review, awaiting_answer = _active_review_awaiting(model)
    if active_review:
        model["active_review"] = active_review
    if awaiting_answer and not student_text:
        question = str(active_review.get("question") or "").strip()
        if not question:
            question = "Please answer the pending review question in your own words."
        return {
            "messages": _append_turn_messages(student_text, f"You still have a pending review question:\n{question}"),
            "consumed_student_text": student_text,
            "student_answer": "",
            "student_model": model,
        }

    planned = state.get("planned_review_target") or {}
    requested_target: Dict[str, Any] | None = None
    if isinstance(planned, dict) and planned.get("concept_id"):
        target = planned
    else:
        if explicit_review_request:
            requested_target = _infer_requested_review_target(
                student_text=student_text,
                model=model,
                current_topic=current_topic,
            )
        if requested_target:
            target = requested_target
        else:
            target = _pick_review_target(model, current_topic=current_topic)
    if not target:
        if explicit_review_request:
            msg = "Tell me what concept you want to review (for example: review self-attention), and I'll start with a quick baseline check."
        else:
            msg = "No due review targets right now. Ask to review a specific concept, or continue learning and I'll queue reviews automatically."
        return {
            "messages": _append_turn_messages(student_text, msg),
            "consumed_student_text": student_text,
            "student_answer": "",
            "student_model": model,
        }

    topic = canonicalize_topic_label(str(target.get("topic") or current_topic or "General"))
    concept_id = str(target.get("concept_id") or "concept")
    label = str(target.get("label") or concept_id)
    proficiency = _clamp01(target.get("proficiency", _concept_proficiency(model, topic, concept_id)))

    prompt = REVIEW_QUESTION_PROMPT.format(
        topic=topic,
        concept_label=label,
        proficiency=f"{int(proficiency * 100)}%",
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        question = (response.content or "").strip()
    except Exception:
        question = ""

    if not question:
        question = f"Quick review: can you explain {label} in your own words and give one example?"

    active_review = {
        "status": "awaiting_answer",
        "awaiting_answer": True,
        "topic": topic,
        "concept_id": concept_id,
        "label": label,
        "question": question,
        "asked_at": _utc_now_iso(),
    }
    model["active_review"] = active_review

    concept_updates: List[Dict[str, Any]] = []
    if requested_target:
        concept_updates.append(
            {
                "topic": topic,
                "concept_id": concept_id,
                "label": label,
                "proficiency_delta": 0.02,
                "evidence": {
                    "kind": "asked",
                    "text": f"Student requested review for {label}",
                    "source": "student",
                    "reasoning": "explicit review request should initialize a review target",
                },
            }
        )

    if requested_target and not _has_review_item(model, concept_id=concept_id, topic=topic):
        review_text = f"Let's start with a quick baseline check for {label}.\n\nReview check:\n{question}"
    else:
        review_text = f"Review check:\n{question}"

    return {
        "messages": _append_turn_messages(student_text, review_text),
        "consumed_student_text": student_text,
        "student_answer": "",
        "student_model": model,
        "current_topic": topic,
        "concept_updates": concept_updates,
    }


def grade_review_answer(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    active_review, awaiting_answer = _active_review_awaiting(model)
    if active_review:
        model["active_review"] = active_review
    if not awaiting_answer:
        return {
            "messages": _append_turn_messages(student_text, "There is no pending review question right now."),
            "consumed_student_text": student_text,
            "student_answer": "",
            "review_result": {},
            "concept_updates": [],
        }

    topic = canonicalize_topic_label(str(active_review.get("topic") or state.get("current_topic") or "General"))
    concept_id = str(active_review.get("concept_id") or "concept")
    label = str(active_review.get("label") or concept_id)
    question = str(active_review.get("question") or "")

    prompt = REVIEW_GRADER_PROMPT.format(
        topic=topic,
        concept_label=label,
        question=question,
        student_answer=student_text or "(empty)",
    )

    score = 0
    feedback = "I couldn't confidently grade that answer, so let's review the concept once more."
    graded_with_model = False
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
        score = int(float(data.get("score", 0)))
        feedback_candidate = data.get("feedback")
        if isinstance(feedback_candidate, str) and feedback_candidate.strip():
            feedback = feedback_candidate.strip()
        graded_with_model = True
    except Exception:
        graded_with_model = False

    floor_score = _minimum_score_from_overlap(label=label, question=question, answer=student_text)
    if not graded_with_model:
        if student_text:
            score = max(score, floor_score, 25)
            feedback = (
                "I couldn't reliably auto-grade this response, so I'm using a conservative score and we'll "
                "keep refining your understanding together."
            )
        else:
            score = 0
            feedback = "I didn't receive an answer yet. Give it a try and I'll coach from there."
    elif floor_score > score:
        score = floor_score
        if score >= 30:
            feedback = (
                "You showed partial understanding, but the explanation is still missing key details. "
                "Let's tighten the definition and one concrete example."
            )

    score = max(0, min(100, score))
    delta = _delta_from_score(score)
    if not graded_with_model and student_text and delta < 0:
        delta = 0.0
    quality = _quality_from_score(score)
    if not graded_with_model and student_text and quality < 2:
        quality = 2
    evidence_kind = "correct_explanation" if score >= 70 else "incorrect"
    if not graded_with_model and student_text and score < 70:
        evidence_kind = "observation"

    concept_update = {
        "topic": topic,
        "concept_id": concept_id,
        "label": label,
        "proficiency_delta": delta,
        "evidence": {
            "kind": evidence_kind,
            "text": student_text[:180],
            "source": "student",
            "reasoning": f"review score={score}",
        },
    }

    active_review = _set_active_review_status(active_review, "completed")
    active_review["last_score"] = score
    active_review["last_answer_at"] = _utc_now_iso()

    mode = str(active_review.get("mode") or "")
    goal_concept = str(active_review.get("goal_concept") or "").replace("_", " ").strip()
    if mode == "prereq_gate" and goal_concept:
        if score >= 70:
            active_review = {}
            next_step = f"Nice. You have enough foundation to continue with {goal_concept}."
        else:
            next_step = (
                f"We should reinforce {label} first, then return to {goal_concept}. "
                f"Ask me to teach {label} step-by-step."
            )
    else:
        next_step = (
            "Great work. Want another review question or a harder application example?"
            if score >= 75
            else "Let's reinforce this. Ask me for a focused explanation or another review question."
        )
    model["active_review"] = active_review
    reply = f"Review score: {score}/100\n{feedback}\n\n{next_step}"

    return {
        "messages": _append_turn_messages(student_text, reply),
        "consumed_student_text": student_text,
        "student_answer": "",
        "student_model": model,
        "concept_updates": [concept_update],
        "review_result": {
            "topic": topic,
            "concept_id": concept_id,
            "label": label,
            "score": score,
            "quality": quality,
        },
    }


def chat(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    student_text = (state.get("student_answer") or "").strip()
    tutor_hint = (state.get("tutor_hint") or "").strip()

    snapshot = model_snapshot(model, current_topic=current_topic)
    review_state = review_snapshot(model, current_topic=current_topic, limit=3)

    system_msg = CHAT_PROMPT.format(
        current_topic=current_topic or "General",
        student_snapshot=snapshot if snapshot else "(no learner data yet)",
        review_status=review_state,
        tutor_hint=tutor_hint if tutor_hint else "(no additional hint)",
    )

    llm_messages: List[Any] = [{"role": "system", "content": system_msg}, *(state.get("messages") or [])]
    if student_text:
        llm_messages.append(HumanMessage(content=student_text))

    try:
        response = llm.invoke(llm_messages)
        ai_text = (response.content or "").strip()
    except Exception:
        ai_text = "I hit a temporary model error. Ask again and I'll continue from your current progress."

    if not ai_text:
        ai_text = "I don't have a useful response yet. Ask me to explain one concept from your current topic."

    return {
        "messages": _append_turn_messages(student_text, ai_text),
        "consumed_student_text": student_text,
        "student_answer": "",
    }


def observe_learning(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    route = state.get("route")
    if not isinstance(route, dict):
        route = {}
    route_intent = str(route.get("intent") or "chat")
    consumed_student_text = (state.get("consumed_student_text") or "").strip()
    messages = state.get("messages") or []
    last_user, last_ai = _last_user_and_ai(messages)

    existing_updates: List[Dict[str, Any]] = []
    raw_existing = state.get("concept_updates")
    if isinstance(raw_existing, list):
        for item in raw_existing:
            if isinstance(item, dict):
                existing_updates.append(dict(item))

    if not consumed_student_text:
        return {"concept_updates": existing_updates, "tutor_hint": ""}
    if not last_user and not last_ai:
        return {"concept_updates": existing_updates, "tutor_hint": ""}

    topics = model.get("topics", {})
    topic_concepts = {}
    if isinstance(topics, dict) and current_topic in topics and isinstance(topics.get(current_topic), dict):
        topic_concepts = topics[current_topic].get("concepts", {}) or {}

    mentioned = _concepts_mentioned_in_text({"concepts": topic_concepts}, f"{last_user}\n{last_ai}")
    related = {concept_id: topic_concepts.get(concept_id) for concept_id in mentioned if concept_id in topic_concepts}

    prompt = WATCHER_PROMPT.format(
        current_topic=current_topic or "General",
        student_message=last_user or "(none)",
        tutor_reply=last_ai or "(none)",
        related_concepts_json=json.dumps(related, ensure_ascii=False, indent=2, sort_keys=True),
    )

    try:
        response = llm.invoke([{"role": "system", "content": prompt}])
        data = extract_json(response.content)
    except Exception:
        data = {"current_topic": current_topic or "General", "tutor_hint": "", "concept_updates": []}

    new_topic = data.get("current_topic", current_topic or "General")
    if not isinstance(new_topic, str) or not new_topic.strip():
        new_topic = current_topic or "General"

    tutor_hint = data.get("tutor_hint", "")
    if not isinstance(tutor_hint, str):
        tutor_hint = ""
    tutor_hint = tutor_hint.strip()
    if len(tutor_hint) > 500:
        tutor_hint = tutor_hint[:500].rstrip()

    observed_updates: List[Dict[str, Any]] = []
    raw_updates = data.get("concept_updates")
    if isinstance(raw_updates, list):
        for item in raw_updates:
            if isinstance(item, dict):
                item = dict(item)
                item["topic"] = str(item.get("topic") or new_topic or current_topic or "General").strip() or "General"
                item["proficiency_delta"] = _fallback_delta_for_update(item)
                observed_updates.append(item)

    if not observed_updates and isinstance(last_user, str) and last_user.strip():
        lowered = _norm_text(last_user)
        inferred_delta = 0.0
        inferred_kind = "observation"
        if any(phrase in lowered for phrase in ("i think i get", "i get it", "makes sense", "got it")):
            inferred_delta = 0.06
            inferred_kind = "correct_explanation"
        elif any(phrase in lowered for phrase in ("confused", "dont get", "don't get", "not sure", "lost")):
            inferred_delta = -0.08
            inferred_kind = "confused"

        if abs(inferred_delta) >= 0.01 and new_topic and new_topic != "General":
            observed_updates.append(
                {
                    "topic": new_topic,
                    "concept_id": _to_concept_id(new_topic),
                    "label": new_topic.replace("_", " ").title(),
                    "proficiency_delta": inferred_delta,
                    "evidence": {
                        "kind": inferred_kind,
                        "text": last_user[:180],
                        "source": "student",
                        "reasoning": "heuristic fallback from student self-report",
                    },
                }
            )

    concept_updates = [*existing_updates, *observed_updates]
    student_turn_text = last_user if isinstance(last_user, str) else ""
    allow_llm_extracted = route_intent in {"learn_request", "pending_switch", "review_request", "pending_answer"}
    llm_extracted: List[Dict[str, Any]] = []
    if allow_llm_extracted:
        llm_extracted = _extract_llm_concept_updates(
            model=model,
            student_text=student_turn_text,
            current_topic=new_topic,
            existing_updates=concept_updates,
        )
    concept_updates.extend(llm_extracted)

    sanitized_updates: List[Dict[str, Any]] = []
    for item in concept_updates:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        if not topic:
            continue
        row = dict(item)
        row["topic"] = topic
        row["proficiency_delta"] = max(-0.2, min(0.2, _fallback_delta_for_update(row)))
        sanitized_updates.append(row)
        if len(sanitized_updates) >= 8:
            break

    return {
        "current_topic": new_topic,
        "tutor_hint": tutor_hint,
        "concept_updates": sanitized_updates,
    }


def assess_understanding(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = canonicalize_topic_label(state.get("current_topic") or "General")
    consumed_student_text = (state.get("consumed_student_text") or "").strip()
    route = state.get("route")
    if not isinstance(route, dict):
        route = {}
    route_intent = str(route.get("intent") or "chat")
    route_topic_raw = str(route.get("target_topic") or "").strip()
    route_topic = canonicalize_topic_label(route_topic_raw) if route_topic_raw else ""
    pedagogical_plan = state.get("pedagogical_plan")
    if not isinstance(pedagogical_plan, dict):
        pedagogical_plan = {}

    existing_updates: List[Dict[str, Any]] = []
    raw_existing = state.get("concept_updates")
    if isinstance(raw_existing, list):
        for item in raw_existing:
            if isinstance(item, dict):
                existing_updates.append(dict(item))

    if not consumed_student_text:
        return {
            "current_topic": current_topic,
            "concept_updates": existing_updates,
            "tutor_hint": "",
            "understanding_assessment": {},
        }

    no_learning_intents = {"session_start", "progress", "pending_social", "pending_defer"}
    if route_intent in no_learning_intents:
        return {
            "current_topic": current_topic,
            "concept_updates": existing_updates,
            "tutor_hint": "",
            "understanding_assessment": {
                "understanding": "unknown",
                "misconception": "",
                "topic": current_topic,
            },
        }
    if route_intent == "chat" and _is_social_message(consumed_student_text) and not _has_explicit_task_intent(consumed_student_text):
        return {
            "current_topic": current_topic,
            "concept_updates": existing_updates,
            "tutor_hint": "",
            "understanding_assessment": {
                "understanding": "unknown",
                "misconception": "",
                "topic": current_topic,
            },
        }

    messages = state.get("messages") or []
    last_user, last_ai = _last_user_and_ai(messages)
    if not last_user and not last_ai:
        return {
            "current_topic": current_topic,
            "concept_updates": existing_updates,
            "tutor_hint": "",
            "understanding_assessment": {},
        }

    seed_topic = route_topic or current_topic or "General"
    topics = model.get("topics", {})
    topic_concepts = {}
    if isinstance(topics, dict) and seed_topic in topics and isinstance(topics.get(seed_topic), dict):
        topic_concepts = topics[seed_topic].get("concepts", {}) or {}

    mentioned = _concepts_mentioned_in_text({"concepts": topic_concepts}, f"{last_user}\n{last_ai}")
    related = {concept_id: topic_concepts.get(concept_id) for concept_id in mentioned if concept_id in topic_concepts}

    prompt = UNDERSTANDING_ASSESSMENT_PROMPT.format(
        current_topic=seed_topic,
        route_intent=route_intent,
        student_message=last_user or "(none)",
        tutor_reply=last_ai or "(none)",
        pedagogical_plan_json=json.dumps(pedagogical_plan, ensure_ascii=False, indent=2, sort_keys=True),
        related_concepts_json=json.dumps(related, ensure_ascii=False, indent=2, sort_keys=True),
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
    except Exception:
        data = {}

    understanding = str(data.get("understanding") or "").strip().lower()
    if understanding not in {"unknown", "confused", "partial", "solid"}:
        understanding = "unknown"
    misconception = str(data.get("misconception") or "").strip()

    tutor_hint = str(data.get("tutor_hint") or "").strip()
    if len(tutor_hint) > 500:
        tutor_hint = tutor_hint[:500].rstrip()

    candidate_topic_raw = str(data.get("topic") or "").strip()
    candidate_topic = canonicalize_topic_label(candidate_topic_raw) if candidate_topic_raw else ""
    if route_topic:
        new_topic = route_topic
    elif candidate_topic and route_intent in {"learn_request", "pending_switch"}:
        new_topic = candidate_topic
    elif candidate_topic and route_intent in {"review_request", "pending_answer"} and _topic_exists(model, candidate_topic):
        new_topic = candidate_topic
    else:
        new_topic = current_topic or "General"

    observed_updates: List[Dict[str, Any]] = []
    raw_updates = data.get("concept_updates")
    if isinstance(raw_updates, list):
        for item in raw_updates:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row_topic_raw = str(row.get("topic") or "").strip()
            row_topic = canonicalize_topic_label(row_topic_raw) if row_topic_raw else new_topic
            if not row_topic:
                continue
            if row_topic != new_topic:
                keep_row_topic = (
                    route_intent in {"review_request", "pending_answer"}
                    and _topic_exists(model, row_topic)
                )
                if not keep_row_topic:
                    row_topic = new_topic
            row["topic"] = row_topic
            row["proficiency_delta"] = _fallback_delta_for_update(row)
            observed_updates.append(row)

    if not observed_updates and understanding == "confused" and isinstance(last_user, str) and last_user.strip():
        fallback_concept = _to_concept_id(new_topic)
        observed_updates.append(
            {
                "topic": new_topic,
                "concept_id": fallback_concept,
                "label": new_topic.replace("_", " ").title(),
                "proficiency_delta": -0.08,
                "evidence": {
                    "kind": "confused",
                    "text": last_user[:180],
                    "source": "student",
                    "reasoning": "student signaled confusion in this turn",
                },
            }
        )

    concept_updates = [*existing_updates, *observed_updates]
    student_turn_text = last_user if isinstance(last_user, str) else ""
    llm_extracted = _extract_llm_concept_updates(
        model=model,
        student_text=student_turn_text,
        current_topic=new_topic,
        existing_updates=concept_updates,
    )
    concept_updates.extend(llm_extracted)

    sanitized_updates: List[Dict[str, Any]] = []
    for item in concept_updates:
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic") or "").strip()
        if not topic:
            continue
        row = dict(item)
        row["topic"] = topic
        row["proficiency_delta"] = max(-0.2, min(0.2, _fallback_delta_for_update(row)))
        sanitized_updates.append(row)
        if len(sanitized_updates) >= 8:
            break

    filtered_updates = _filter_concept_updates_for_persistence(
        model=model,
        updates=sanitized_updates,
        route_intent=route_intent,
        default_topic=new_topic,
    )

    return {
        "current_topic": new_topic,
        "tutor_hint": tutor_hint,
        "concept_updates": filtered_updates,
        "understanding_assessment": {
            "understanding": understanding,
            "misconception": misconception,
            "topic": new_topic,
        },
    }


def update_student_model(state: TutorState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id") or "default"

    model = ensure_model_shape(state.get("student_model") or {})
    topic = canonicalize_topic_label(state.get("current_topic") or "General")
    model.setdefault("active_review", {})
    model.setdefault("meta", {"turn_count": 0})
    raw_updates = state.get("concept_updates") or []
    route = state.get("route")
    if not isinstance(route, dict):
        route = {}
    route_intent = str(route.get("intent") or "")
    concept_updates = _filter_concept_updates_for_persistence(
        model=model,
        updates=raw_updates if isinstance(raw_updates, list) else [],
        route_intent=route_intent,
        default_topic=topic,
    )

    if isinstance(concept_updates, list) and concept_updates:
        model = _merge_concept_updates(model=model, updates=concept_updates, current_topic=topic)

    review_result = state.get("review_result")
    if isinstance(review_result, dict) and review_result:
        concept_id = str(review_result.get("concept_id") or "concept")
        label = str(review_result.get("label") or concept_id)
        result_topic = canonicalize_topic_label(str(review_result.get("topic") or topic))
        quality = int(review_result.get("quality", 3))
        prof = _concept_proficiency(model, result_topic, concept_id)
        model = update_review_queue(
            model,
            topic=result_topic,
            concept_id=concept_id,
            label=label,
            quality=quality,
            proficiency=prof,
        )
    elif isinstance(concept_updates, list):
        for update in concept_updates:
            if not isinstance(update, dict):
                continue
            update_topic = canonicalize_topic_label(str(update.get("topic") or topic))
            concept_id = str(update.get("concept_id") or "")
            label = str(update.get("label") or concept_id or "")
            if not concept_id or not label:
                continue
            try:
                delta = float(update.get("proficiency_delta", 0.0))
            except (TypeError, ValueError):
                delta = 0.0
            should_seed = not _has_review_item(model, concept_id=concept_id, topic=update_topic)
            if abs(delta) < 0.03 and not should_seed:
                continue
            prof = _concept_proficiency(model, update_topic, concept_id)
            model = update_review_queue(
                model,
                topic=update_topic,
                concept_id=concept_id,
                label=label,
                quality=_quality_from_delta(delta),
                proficiency=prof,
            )

    consumed = (state.get("consumed_student_text") or "").strip()
    _, last_ai = _last_user_and_ai(state.get("messages") or [])
    turn_log = model.get("turn_log")
    if not isinstance(turn_log, list):
        turn_log = []
    if consumed or last_ai:
        turn_log.append({"user": consumed, "assistant": last_ai})
    model["turn_log"] = turn_log[-80:]

    meta = model.get("meta")
    if not isinstance(meta, dict):
        meta = {"turn_count": 0}
    try:
        turn_count = int(meta.get("turn_count", 0))
    except (TypeError, ValueError):
        turn_count = 0
    meta["turn_count"] = turn_count + 1
    route_data = state.get("route")
    if not isinstance(route_data, dict):
        route_data = {}
    active_review, _ = _active_review_awaiting(model)
    if active_review:
        model["active_review"] = active_review
    active_status = str(active_review.get("status") or "none") if active_review else "none"
    route_log = meta.get("route_log")
    if not isinstance(route_log, list):
        route_log = []
    route_log.append(
        {
            "ts": _utc_now_iso(),
            "intent": str(route_data.get("intent") or ""),
            "reason": str(route_data.get("reason") or state.get("action_reason") or ""),
            "confidence": _clamp01(route_data.get("confidence", 0.0)),
            "pause_pending": bool(route_data.get("pause_pending")),
            "active_review_status": active_status,
        }
    )
    meta["route_log"] = route_log[-200:]
    model["meta"] = meta

    hint = (state.get("tutor_hint") or "").strip()
    if hint:
        model["last_tutor_hint"] = hint

    store_save_student_model(student_id=str(thread_id), model=model)
    return {
        "student_model": model,
        "concept_updates": [],
        "review_result": {},
        "planned_review_target": {},
        "planned_goal_concept": "",
        "pedagogical_plan": {},
        "understanding_assessment": {},
        "consumed_student_text": "",
    }


def _create_legacy_tutor_graph():
    graph = StateGraph(TutorState)

    graph.add_node("load_student_model", load_student_model)
    graph.add_node("router", router)
    graph.add_node("plan_next_step", plan_next_step)
    graph.add_node("greet", greet)
    graph.add_node("progress_report", progress_report)
    graph.add_node("ask_prereq_diagnostic", ask_prereq_diagnostic)
    graph.add_node("teach_foundation", teach_foundation)
    graph.add_node("remind_review_question", remind_review_question)
    graph.add_node("ask_review_question", ask_review_question)
    graph.add_node("grade_review_answer", grade_review_answer)
    graph.add_node("chat", chat)
    graph.add_node("observe_learning", observe_learning)
    graph.add_node("update_student_model", update_student_model)

    graph.set_entry_point("load_student_model")
    graph.add_edge("load_student_model", "router")

    graph.add_conditional_edges(
        "router",
        lambda state: state["next"],
        {
            "chat": "chat",
            "greet": "greet",
            "progress_report": "progress_report",
            "ask_review_question": "ask_review_question",
            "remind_review_question": "remind_review_question",
            "grade_review_answer": "grade_review_answer",
            "plan_next_step": "plan_next_step",
            "teach_foundation": "teach_foundation",
        },
    )

    graph.add_conditional_edges(
        "plan_next_step",
        lambda state: state["next"],
        {
            "ask_prereq_diagnostic": "ask_prereq_diagnostic",
            "ask_review_question": "ask_review_question",
            "chat": "chat",
        },
    )

    graph.add_edge("greet", "update_student_model")
    graph.add_edge("progress_report", "update_student_model")
    graph.add_edge("ask_prereq_diagnostic", "update_student_model")
    graph.add_edge("teach_foundation", "update_student_model")
    graph.add_edge("remind_review_question", "update_student_model")
    graph.add_edge("ask_review_question", "update_student_model")
    graph.add_edge("grade_review_answer", "observe_learning")
    graph.add_edge("chat", "observe_learning")
    graph.add_edge("observe_learning", "update_student_model")
    graph.add_edge("update_student_model", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


def create_tutor_graph():
    if not TUTOR_ROUTER_V2:
        return _create_legacy_tutor_graph()

    graph = StateGraph(TutorState)
    graph.add_node("load_student_model", load_student_model)
    graph.add_node("route_turn", route_turn)
    graph.add_node("pedagogical_plan", pedagogical_plan)
    graph.add_node("respond_turn", respond_turn)
    graph.add_node("assess_understanding", assess_understanding)
    graph.add_node("update_student_model", update_student_model)

    graph.set_entry_point("load_student_model")
    graph.add_edge("load_student_model", "route_turn")
    graph.add_edge("route_turn", "pedagogical_plan")
    graph.add_edge("pedagogical_plan", "respond_turn")
    graph.add_edge("respond_turn", "assess_understanding")
    graph.add_edge("assess_understanding", "update_student_model")
    graph.add_edge("update_student_model", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


app = create_tutor_graph()


if __name__ == "__main__":
    print("Tutor agent loaded")
    print("Graph nodes:", app.get_graph().nodes.keys())
