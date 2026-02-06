"""A tiny JSON-backed external store for student models.

Why this exists:
- LangGraph state (with checkpointers) is great for in-graph memory.
- Sometimes you also want an *external* store that persists and can be inspected.

Design goals:
- Extremely small surface area
- Human-readable JSON on disk
- Keyed by `student_id` (we'll use LangGraph's `thread_id`)

This is intentionally not a database. It's a learning scaffold.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any


_STORE_PATH = os.path.join(os.path.dirname(__file__), "student_store.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_all() -> dict[str, Any]:
    if not os.path.exists(_STORE_PATH):
        return {}
    try:
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_all(data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
    tmp = _STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, _STORE_PATH)


def default_student_model() -> dict[str, Any]:
    """Create a fresh multi-topic student model."""
    return {
        "topics": {},  # Structure: {topic_name: {concepts: {}, last_studied: str, total_practice: int}}
        "review_queue": [],
        "active_review": {},
        "meta": {"turn_count": 0},
        "turn_log": [],
        "updated_at": _utc_now_iso(),
    }


def ensure_model_shape(model: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a student model so downstream nodes can rely on its shape."""
    safe = dict(model or {})

    topics = safe.get("topics")
    if not isinstance(topics, dict):
        topics = {}
    safe["topics"] = topics

    turn_log = safe.get("turn_log")
    if not isinstance(turn_log, list):
        turn_log = []
    safe["turn_log"] = turn_log

    review_queue = safe.get("review_queue")
    if not isinstance(review_queue, list):
        review_queue = []
    safe["review_queue"] = review_queue

    active_review = safe.get("active_review")
    if not isinstance(active_review, dict):
        active_review = {}
    safe["active_review"] = active_review

    meta = safe.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    turn_count = meta.get("turn_count")
    if not isinstance(turn_count, int):
        meta["turn_count"] = 0
    safe["meta"] = meta

    if not isinstance(safe.get("updated_at"), str):
        safe["updated_at"] = _utc_now_iso()

    return safe


def load_student_model(*, student_id: str) -> dict[str, Any]:
    """Load a student's model from disk, or create one if missing."""
    all_data = _read_all()
    model = all_data.get(student_id)
    if not isinstance(model, dict):
        model = default_student_model()
    return ensure_model_shape(model)


def save_student_model(*, student_id: str, model: dict[str, Any]) -> None:
    """Persist a student's model to disk."""
    all_data = _read_all()
    model = ensure_model_shape(model)
    model["updated_at"] = _utc_now_iso()
    all_data[student_id] = model
    _write_all(all_data)


def get_due_reviews(
    model: dict[str, Any],
    *,
    current_topic: str = "",
    limit: int = 5,
    now_iso: str | None = None,
) -> list[dict[str, Any]]:
    """Return due review items sorted by due time then weakest proficiency."""
    safe = ensure_model_shape(model)
    now_dt = _parse_iso(now_iso) or datetime.now(timezone.utc)
    topic_filter = (current_topic or "").strip().lower()

    due_items: list[dict[str, Any]] = []
    for raw in safe.get("review_queue", []):
        if not isinstance(raw, dict):
            continue
        topic = str(raw.get("topic") or "")
        if topic_filter and topic.lower() != topic_filter:
            continue

        due_dt = _parse_iso(raw.get("due_at"))
        if due_dt is None:
            continue
        if due_dt > now_dt:
            continue

        item = dict(raw)
        item["_due_dt"] = due_dt
        try:
            item["_prof"] = float(item.get("proficiency", 0.0))
        except (TypeError, ValueError):
            item["_prof"] = 0.0
        due_items.append(item)

    due_items.sort(key=lambda x: (x["_due_dt"], x["_prof"]))
    cleaned: list[dict[str, Any]] = []
    for item in due_items[: max(0, int(limit))]:
        item.pop("_due_dt", None)
        item.pop("_prof", None)
        cleaned.append(item)
    return cleaned


def update_review_queue(
    model: dict[str, Any],
    *,
    topic: str,
    concept_id: str,
    label: str,
    quality: int,
    proficiency: float,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Apply an SM-2-like schedule update for one concept review item."""
    safe = ensure_model_shape(model)
    queue = safe.get("review_queue", [])
    now_dt = _parse_iso(now_iso) or datetime.now(timezone.utc)

    topic_norm = (topic or "General").strip() or "General"
    concept_norm = (concept_id or "concept").strip() or "concept"
    label_norm = (label or concept_norm).strip() or concept_norm
    quality = max(0, min(5, int(quality)))
    proficiency = max(0.0, min(1.0, float(proficiency)))

    target_index = None
    for idx, item in enumerate(queue):
        if not isinstance(item, dict):
            continue
        if item.get("topic") == topic_norm and item.get("concept_id") == concept_norm:
            target_index = idx
            break

    if target_index is None:
        item = {
            "topic": topic_norm,
            "concept_id": concept_norm,
            "label": label_norm,
            "ease": 2.5,
            "repetitions": 0,
            "interval_days": 1,
            "due_at": now_dt.isoformat(timespec="seconds"),
            "last_reviewed": None,
            "last_quality": None,
            "proficiency": proficiency,
        }
        queue.append(item)
        target_index = len(queue) - 1
    else:
        item = queue[target_index]

    ease = item.get("ease", 2.5)
    repetitions = item.get("repetitions", 0)
    interval_days = item.get("interval_days", 1)
    try:
        ease = float(ease)
    except (TypeError, ValueError):
        ease = 2.5
    try:
        repetitions = int(repetitions)
    except (TypeError, ValueError):
        repetitions = 0
    try:
        interval_days = int(interval_days)
    except (TypeError, ValueError):
        interval_days = 1

    if quality < 3:
        repetitions = 0
        interval_days = 1
    else:
        repetitions += 1
        if repetitions == 1:
            interval_days = 1
        elif repetitions == 2:
            interval_days = 3
        else:
            interval_days = max(1, round(interval_days * ease))

    penalty = (5 - quality) * (0.08 + (5 - quality) * 0.02)
    ease = max(1.3, ease + (0.1 - penalty))
    due_dt = now_dt + timedelta(days=interval_days)

    item["topic"] = topic_norm
    item["concept_id"] = concept_norm
    item["label"] = label_norm
    item["ease"] = round(ease, 3)
    item["repetitions"] = repetitions
    item["interval_days"] = interval_days
    item["due_at"] = due_dt.isoformat(timespec="seconds")
    item["last_reviewed"] = now_dt.isoformat(timespec="seconds")
    item["last_quality"] = quality
    item["proficiency"] = proficiency
    queue[target_index] = item

    safe["review_queue"] = queue
    return safe


def review_snapshot(
    model: dict[str, Any],
    *,
    current_topic: str = "",
    limit: int = 5,
    now_iso: str | None = None,
) -> str:
    """Human-readable review queue summary for planning/reporting."""
    safe = ensure_model_shape(model)
    queue = [x for x in safe.get("review_queue", []) if isinstance(x, dict)]
    if not queue:
        return "Review queue: no items yet."

    now_dt = _parse_iso(now_iso) or datetime.now(timezone.utc)
    topic_filter = (current_topic or "").strip().lower()
    filtered = []
    for item in queue:
        if topic_filter:
            topic = str(item.get("topic") or "")
            if topic.lower() != topic_filter:
                continue
        filtered.append(item)

    due = get_due_reviews(
        safe,
        current_topic=current_topic,
        limit=limit,
        now_iso=now_dt.isoformat(timespec="seconds"),
    )
    total = len(filtered)
    due_count = len(
        get_due_reviews(
            safe,
            current_topic=current_topic,
            limit=max(1, total),
            now_iso=now_dt.isoformat(timespec="seconds"),
        )
    )

    if not due:
        scope = current_topic if current_topic and current_topic != "General" else "all topics"
        return f"Review queue ({scope}): {due_count}/{total} due now. No due cards in top {limit}."

    lines = []
    for item in due:
        label = item.get("label") or item.get("concept_id") or "concept"
        topic = item.get("topic") or "General"
        days = item.get("interval_days", 1)
        lines.append(f"{label} [{topic}] (interval {days}d)")

    scope = current_topic if current_topic and current_topic != "General" else "all topics"
    return (
        f"Review queue ({scope}): {due_count}/{total} due now.\n"
        f"Due next: {', '.join(lines)}"
    )


def model_snapshot(
    model: dict[str, Any], 
    *, 
    current_topic: str = "",
    max_concepts: int = 8
) -> str:
    """Return an actionable snapshot grouped by mastery level.
    
    Args:
        model: The student model dict
        current_topic: If provided, show only this topic. If empty/General, show cross-topic overview
        max_concepts: Max concepts to show per category
    """

    topics = model.get("topics", {})
    
    # If specific topic requested and exists
    if current_topic and current_topic != "General" and current_topic in topics:
        topic_data = topics.get(current_topic, {})
        concepts = topic_data.get("concepts", {})
        
        if not isinstance(concepts, dict) or not concepts:
            return f"Topic '{current_topic}': No concepts learned yet"
        
        # Group by mastery within this topic
        struggling, developing, proficient = [], [], []
        
        for cid, meta in concepts.items():
            if not isinstance(cid, str) or not isinstance(meta, dict):
                continue
            
            prof = max(0.0, min(1.0, float(meta.get("proficiency", 0.0))))
            item = {
                "id": cid,
                "label": meta.get("label", cid),
                "prof": prof,
                "mastery": meta.get("mastery_level", "introduced"),
                "trajectory": meta.get("trajectory", "stable"),
                "last_kind": "",
            }
            evidence = meta.get("evidence", [])
            if evidence and isinstance(evidence, list):
                item["last_kind"] = evidence[-1].get("kind", "unknown") if evidence else "unknown"
            
            if prof < 0.3:
                struggling.append(item)
            elif prof < 0.6:
                developing.append(item)
            else:
                proficient.append(item)
        
        parts = []
        if struggling:
            struggling.sort(key=lambda x: x["prof"])
            items_str = ", ".join(
                f"{c['label']} ({int(c['prof']*100)}%, {c['trajectory']}, last: {c['last_kind']})"
                for c in struggling[:max_concepts]
            )
            parts.append(f"STRUGGLING ({len(struggling)}): {items_str}")
        
        if developing:
            developing.sort(key=lambda x: x["prof"])
            items_str = ", ".join(
                f"{c['label']} ({int(c['prof']*100)}%, {c['trajectory']})"
                for c in developing[:max_concepts]
            )
            parts.append(f"DEVELOPING ({len(developing)}): {items_str}")
        
        if proficient:
            proficient.sort(key=lambda x: -x["prof"])
            items_str = ", ".join(
                f"{c['label']} ({int(c['prof']*100)}%)"
                for c in proficient[:2]
            )
            parts.append(f"PROFICIENT ({len(proficient)}): {items_str}")
        
        result = f"Topic: {current_topic}\n" + "\n".join(parts) if parts else f"Topic: {current_topic} (no concepts yet)"
        return result
    
    # Cross-topic overview
    all_concepts = []
    for topic_name, topic_data in topics.items():
        if not isinstance(topic_data, dict):
            continue
        for cid, meta in topic_data.get("concepts", {}).items():
            if not isinstance(meta, dict):
                continue
            prof = max(0.0, min(1.0, float(meta.get("proficiency", 0.0))))
            all_concepts.append({
                "id": cid,
                "topic": topic_name,
                "label": meta.get("label", cid),
                "prof": prof,
                "mastery": meta.get("mastery_level", "introduced"),
                "trajectory": meta.get("trajectory", "stable"),
            })
    
    if not all_concepts:
        return "No concepts learned yet across any topic"
    
    # Group by mastery across all topics
    struggling = [c for c in all_concepts if c["prof"] < 0.3]
    developing = [c for c in all_concepts if 0.3 <= c["prof"] < 0.6]
    proficient = [c for c in all_concepts if c["prof"] >= 0.6]
    
    parts = []
    if struggling:
        struggling.sort(key=lambda x: x["prof"])
        items_str = ", ".join(
            f"{c['label']} ({c['topic']}, {int(c['prof']*100)}%)"
            for c in struggling[:max_concepts]
        )
        parts.append(f"CROSS-TOPIC STRUGGLING ({len(struggling)}): {items_str}")
    
    if developing:
        developing.sort(key=lambda x: x["prof"])
        items_str = ", ".join(
            f"{c['label']} ({c['topic']}, {int(c['prof']*100)}%)"
            for c in developing[:max_concepts]
        )
        parts.append(f"CROSS-TOPIC DEVELOPING ({len(developing)}): {items_str}")
    
    if proficient:
        proficient.sort(key=lambda x: -x["prof"])
        items_str = ", ".join(
            f"{c['label']} ({c['topic']}, {int(c['prof']*100)}%)"
            for c in proficient[:3]
        )
        parts.append(f"CROSS-TOPIC PROFICIENT ({len(proficient)}): {items_str}")
    
    return "\n".join(parts)
