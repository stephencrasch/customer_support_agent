"""Adaptive tutor graph with persistent knowledge and scheduled review.

Key design choices:
- Deterministic routing first (progress/review/awaiting-answer guards).
- LLMs for language-heavy tasks only (teaching, observation, grading).
- Deterministic persistence + review scheduling in code.
"""

from __future__ import annotations

import json
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


class TutorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_topic: str
    student_answer: str
    consumed_student_text: str
    student_model: Dict[str, Any]
    tutor_hint: str
    concept_updates: List[Dict[str, Any]]
    review_result: Dict[str, Any]
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


def _clamp01(x: Any) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _collect_concept_labels(model: Dict[str, Any]) -> Dict[str, List[str]]:
    concepts = model.get("concepts")
    if not isinstance(concepts, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for concept_id, meta in concepts.items():
        if not isinstance(concept_id, str) or not isinstance(meta, dict):
            continue
        names: List[str] = []
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


def _merge_concept_updates(
    *,
    model: Dict[str, Any],
    updates: List[Dict[str, Any]],
    current_topic: str,
    evidence_per_concept: int = 12,
) -> Dict[str, Any]:
    model = ensure_model_shape(model)
    topics = model.get("topics", {})
    topic = (current_topic or "General").strip() or "General"

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

    now = _utc_now_iso()

    for update in updates or []:
        if not isinstance(update, dict):
            continue

        label = update.get("label")
        if not isinstance(label, str) or not label.strip():
            continue

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
    topic_data["total_practice"] = practice_total + len(updates or [])
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
1. Answer the student's message clearly.
2. Make one concrete next-step suggestion.
3. Keep it concise (2-4 short paragraphs).
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


PROGRESS_KEYWORDS = (
    "progress",
    "weakest",
    "show my graph",
    "knowledge graph",
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


def _is_progress_request(text: str) -> bool:
    lowered = _norm_text(text)
    return any(keyword in lowered for keyword in PROGRESS_KEYWORDS)


def _is_review_request(text: str) -> bool:
    lowered = _norm_text(text)
    return any(keyword in lowered for keyword in REVIEW_KEYWORDS)


def load_student_model(state: TutorState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id") or "default"
    model = store_load_student_model(student_id=str(thread_id))
    return {
        "student_model": ensure_model_shape(model),
        "concept_updates": [],
        "review_result": {},
        "action_reason": "",
    }


def router(state: TutorState) -> Dict[str, Any]:
    student_text = (state.get("student_answer") or "").strip()
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    active_review = model.get("active_review", {})
    awaiting_answer = bool(active_review.get("awaiting_answer")) if isinstance(active_review, dict) else False

    if awaiting_answer:
        if student_text:
            return {"next": "grade_review_answer", "action_reason": "grading pending review answer"}
        return {"next": "ask_review_question", "action_reason": "reminding pending review question"}

    if not student_text:
        return {"next": "greet", "action_reason": "session kickoff"}

    if _is_progress_request(student_text):
        return {"next": "progress_report", "action_reason": "explicit progress request"}

    if _is_review_request(student_text):
        return {"next": "ask_review_question", "action_reason": "explicit review request"}

    due = get_due_reviews(model, current_topic=current_topic, limit=1)
    turn_count = model.get("meta", {}).get("turn_count", 0) if isinstance(model.get("meta"), dict) else 0
    if due and isinstance(turn_count, int) and turn_count > 0 and turn_count % 4 == 0:
        return {"next": "ask_review_question", "action_reason": "scheduled proactive review"}

    return {"next": "chat", "action_reason": "default tutoring chat"}


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
    current_topic = state.get("current_topic", "")
    student_text = (state.get("student_answer") or "").strip()

    topic_part = model_snapshot(model, current_topic=current_topic or "General")
    review_part = review_snapshot(model, current_topic=current_topic, limit=5)
    msg = f"{topic_part}\n\n{review_part}"

    return {
        "messages": _append_turn_messages(student_text, msg),
        "consumed_student_text": student_text,
        "student_answer": "",
    }


def ask_review_question(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    current_topic = state.get("current_topic", "")
    student_text = (state.get("student_answer") or "").strip()

    target = _pick_review_target(model, current_topic=current_topic)
    if not target:
        msg = (
            "No review targets yet. Ask me to teach a topic first, then I can "
            "start building your review queue."
        )
        return {
            "messages": _append_turn_messages(student_text, msg),
            "consumed_student_text": student_text,
            "student_answer": "",
            "student_model": model,
        }

    topic = str(target.get("topic") or current_topic or "General")
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
        "awaiting_answer": True,
        "topic": topic,
        "concept_id": concept_id,
        "label": label,
        "question": question,
        "asked_at": _utc_now_iso(),
    }
    model["active_review"] = active_review

    return {
        "messages": _append_turn_messages(student_text, f"Review check:\n{question}"),
        "consumed_student_text": student_text,
        "student_answer": "",
        "student_model": model,
    }


def grade_review_answer(state: TutorState) -> Dict[str, Any]:
    model = ensure_model_shape(state.get("student_model") or {})
    student_text = (state.get("student_answer") or "").strip()
    active_review = model.get("active_review")
    if not isinstance(active_review, dict) or not active_review.get("awaiting_answer"):
        return {
            "messages": _append_turn_messages(student_text, "There is no pending review question right now."),
            "consumed_student_text": student_text,
            "student_answer": "",
            "review_result": {},
            "concept_updates": [],
        }

    topic = str(active_review.get("topic") or state.get("current_topic") or "General")
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
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(response.content)
        score = int(float(data.get("score", 0)))
        feedback_candidate = data.get("feedback")
        if isinstance(feedback_candidate, str) and feedback_candidate.strip():
            feedback = feedback_candidate.strip()
    except Exception:
        pass

    score = max(0, min(100, score))
    delta = _delta_from_score(score)
    quality = _quality_from_score(score)

    concept_update = {
        "concept_id": concept_id,
        "label": label,
        "proficiency_delta": delta,
        "evidence": {
            "kind": "correct_explanation" if score >= 70 else "incorrect",
            "text": student_text[:180],
            "source": "student",
            "reasoning": f"review score={score}",
        },
    }

    active_review["awaiting_answer"] = False
    active_review["last_score"] = score
    active_review["last_answer_at"] = _utc_now_iso()
    model["active_review"] = active_review

    next_step = (
        "Great work. Want another review question or a harder application example?"
        if score >= 75
        else "Let's reinforce this. Ask me for a focused explanation or another review question."
    )
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
    messages = state.get("messages") or []
    last_user, last_ai = _last_user_and_ai(messages)
    if not last_user and not last_ai:
        return {"concept_updates": [], "tutor_hint": ""}

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

    concept_updates: List[Dict[str, Any]] = []
    raw_updates = data.get("concept_updates")
    if isinstance(raw_updates, list):
        for item in raw_updates:
            if isinstance(item, dict):
                concept_updates.append(item)

    return {
        "current_topic": new_topic,
        "tutor_hint": tutor_hint,
        "concept_updates": concept_updates,
    }


def update_student_model(state: TutorState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id") or "default"

    model = ensure_model_shape(state.get("student_model") or {})
    topic = (state.get("current_topic") or "General").strip() or "General"
    model.setdefault("active_review", {})
    model.setdefault("meta", {"turn_count": 0})
    concept_updates = state.get("concept_updates") or []

    if isinstance(concept_updates, list) and concept_updates:
        model = _merge_concept_updates(model=model, updates=concept_updates, current_topic=topic)

    review_result = state.get("review_result")
    if isinstance(review_result, dict) and review_result:
        concept_id = str(review_result.get("concept_id") or "concept")
        label = str(review_result.get("label") or concept_id)
        result_topic = str(review_result.get("topic") or topic)
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
            concept_id = str(update.get("concept_id") or "")
            label = str(update.get("label") or concept_id or "")
            if not concept_id or not label:
                continue
            try:
                delta = float(update.get("proficiency_delta", 0.0))
            except (TypeError, ValueError):
                delta = 0.0
            if abs(delta) < 0.05:
                continue
            prof = _concept_proficiency(model, topic, concept_id)
            model = update_review_queue(
                model,
                topic=topic,
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
    model["meta"] = meta

    hint = (state.get("tutor_hint") or "").strip()
    if hint:
        model["last_tutor_hint"] = hint

    store_save_student_model(student_id=str(thread_id), model=model)
    return {
        "student_model": model,
        "concept_updates": [],
        "review_result": {},
        "consumed_student_text": "",
    }


def create_tutor_graph():
    graph = StateGraph(TutorState)

    graph.add_node("load_student_model", load_student_model)
    graph.add_node("router", router)
    graph.add_node("greet", greet)
    graph.add_node("progress_report", progress_report)
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
            "greet": "greet",
            "progress_report": "progress_report",
            "ask_review_question": "ask_review_question",
            "grade_review_answer": "grade_review_answer",
            "chat": "chat",
        },
    )

    graph.add_edge("greet", "update_student_model")
    graph.add_edge("progress_report", "update_student_model")
    graph.add_edge("ask_review_question", "update_student_model")
    graph.add_edge("grade_review_answer", "update_student_model")
    graph.add_edge("chat", "observe_learning")
    graph.add_edge("observe_learning", "update_student_model")
    graph.add_edge("update_student_model", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


app = create_tutor_graph()


if __name__ == "__main__":
    print("Tutor agent loaded")
    print("Graph nodes:", app.get_graph().nodes.keys())
