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
_TOPIC_ACRONYMS = {"llm", "llms", "nlp", "rnn", "rnns", "lstm", "lstms", "gpt", "cnn", "cnns", "api", "apis"}


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


def _topic_norm_key(topic: str) -> str:
    cleaned = " ".join(str(topic or "").replace("_", " ").replace("-", " ").split()).strip().lower()
    if not cleaned:
        return ""

    stopwords = {
        "concept",
        "concepts",
        "topic",
        "topics",
        "overview",
        "intro",
        "introduction",
        "basics",
        "basic",
        "mechanism",
        "mechanisms",
    }
    parts: list[str] = []
    for token in cleaned.split():
        part = "".join(ch for ch in token if ch.isalnum())
        if not part:
            continue
        if part.endswith("s") and len(part) > 3:
            part = part[:-1]
        if part in stopwords:
            continue
        parts.append(part)

    if not parts:
        parts = ["".join(ch for ch in token if ch.isalnum()) for token in cleaned.split()]
        parts = [part for part in parts if part]

    return "".join(parts)


def canonicalize_topic_label(topic: str) -> str:
    cleaned = " ".join(str(topic or "").replace("_", " ").replace("-", " ").split()).strip()
    if not cleaned:
        return "General"

    words: list[str] = []
    for token in cleaned.split():
        lower = token.lower()
        if lower in _TOPIC_ACRONYMS:
            words.append(lower.upper())
        elif token.isupper() and len(token) <= 5:
            words.append(token)
        else:
            words.append(token.capitalize())
    return " ".join(words) or "General"


def _canonical_topic_for_model(topic: str, topics: dict[str, Any]) -> str:
    wanted = _topic_norm_key(topic)
    if not wanted:
        return "General"
    if isinstance(topics, dict):
        for existing in topics.keys():
            if _topic_norm_key(str(existing)) == wanted:
                return str(existing)
    return canonicalize_topic_label(topic)


def _to_concept_id(value: str) -> str:
    cleaned = str(value or "").strip().lower()
    out: list[str] = []
    prev_sep = False
    for ch in cleaned:
        if ch.isalnum():
            out.append(ch)
            prev_sep = False
        elif not prev_sep:
            out.append("_")
            prev_sep = True
    concept_id = "".join(out).strip("_")
    return concept_id or "concept"


def _concept_norm_key(value: str) -> str:
    cleaned = " ".join(str(value or "").replace("_", " ").replace("-", " ").split()).strip().lower()
    if not cleaned:
        return ""

    stopwords = {
        "concept",
        "concepts",
        "mechanism",
        "mechanisms",
        "topic",
        "topics",
        "overview",
        "intro",
        "introduction",
        "basics",
        "basic",
        "theory",
        "fundamental",
        "fundamentals",
    }
    tokens: list[str] = []
    for token in cleaned.split():
        token_clean = "".join(ch for ch in token if ch.isalnum())
        if not token_clean:
            continue
        if token_clean.endswith("s") and len(token_clean) > 3:
            token_clean = token_clean[:-1]
        if token_clean in stopwords:
            continue
        tokens.append(token_clean)
    if not tokens:
        tokens = ["".join(ch for ch in token if ch.isalnum()) for token in cleaned.split()]
        tokens = [token for token in tokens if token]
    return " ".join(tokens)


def _concept_id_quality(concept_id: str) -> tuple[int, int, int]:
    parts = [part for part in str(concept_id or "").split("_") if part]
    stop = {
        "concept",
        "mechanism",
        "topic",
        "overview",
        "intro",
        "introduction",
        "basics",
        "basic",
        "theory",
        "fundamental",
    }
    stop_count = sum(1 for part in parts if part in stop)
    return (stop_count, len(parts), len(str(concept_id or "")))


def _prefer_concept_id(current_id: str, incoming_id: str) -> str:
    if not current_id:
        return incoming_id or "concept"
    if not incoming_id:
        return current_id
    current_quality = _concept_id_quality(current_id)
    incoming_quality = _concept_id_quality(incoming_id)
    if incoming_quality < current_quality:
        return incoming_id
    return current_id


def _clamp01(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return max(0.0, min(1.0, parsed))


def _merge_aliases(*alias_lists: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw in alias_lists:
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str):
                continue
            alias = item.strip()
            if not alias:
                continue
            key = alias.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(alias)
    return merged


def _merge_concept_evidence(existing: Any, incoming: Any, limit: int = 24) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for source in (existing, incoming):
        if not isinstance(source, list):
            continue
        for row in source:
            if isinstance(row, dict):
                merged.append(dict(row))
    if len(merged) <= limit:
        return merged
    return merged[-limit:]


def _merge_concept_meta(base: dict[str, Any], incoming: dict[str, Any], concept_id: str) -> dict[str, Any]:
    merged = dict(base)
    merged.update(incoming)

    base_label = str(base.get("label") or "").strip()
    incoming_label = str(incoming.get("label") or "").strip()
    default_label = concept_id.replace("_", " ").title()
    if not base_label:
        label = incoming_label or default_label
    elif not incoming_label:
        label = base_label
    else:
        label = incoming_label if len(incoming_label) > len(base_label) else base_label
    merged["label"] = label

    base_prof = _clamp01(base.get("proficiency", 0.0))
    incoming_prof = _clamp01(incoming.get("proficiency", 0.0))
    merged["proficiency"] = max(base_prof, incoming_prof)

    base_count = 0
    try:
        base_count = max(0, int(base.get("practice_count", 0)))
    except (TypeError, ValueError):
        base_count = 0
    incoming_count = 0
    try:
        incoming_count = max(0, int(incoming.get("practice_count", 0)))
    except (TypeError, ValueError):
        incoming_count = 0
    merged["practice_count"] = base_count + incoming_count

    base_aliases = base.get("aliases")
    incoming_aliases = incoming.get("aliases")
    label_aliases = [name for name in (base_label, incoming_label) if name and name != label]
    merged["aliases"] = _merge_aliases(
        base_aliases if isinstance(base_aliases, list) else [],
        incoming_aliases if isinstance(incoming_aliases, list) else [],
        label_aliases,
    )

    base_first_seen = _parse_iso(str(base.get("first_seen") or ""))
    incoming_first_seen = _parse_iso(str(incoming.get("first_seen") or ""))
    if base_first_seen and incoming_first_seen:
        first_seen = min(base_first_seen, incoming_first_seen)
    else:
        first_seen = base_first_seen or incoming_first_seen
    if first_seen:
        merged["first_seen"] = first_seen.isoformat(timespec="seconds")

    base_last = _parse_iso(str(base.get("last_practiced") or ""))
    incoming_last = _parse_iso(str(incoming.get("last_practiced") or ""))
    if base_last and incoming_last:
        last_practiced = max(base_last, incoming_last)
    else:
        last_practiced = base_last or incoming_last
    if last_practiced:
        merged["last_practiced"] = last_practiced.isoformat(timespec="seconds")

    merged["evidence"] = _merge_concept_evidence(base.get("evidence"), incoming.get("evidence"))

    notes_candidates = [
        str(base.get("notes") or "").strip(),
        str(incoming.get("notes") or "").strip(),
    ]
    notes_candidates = [note for note in notes_candidates if note]
    if notes_candidates:
        merged["notes"] = max(notes_candidates, key=len)

    if incoming_prof >= base_prof:
        if isinstance(incoming.get("mastery_level"), str):
            merged["mastery_level"] = incoming.get("mastery_level")
        if isinstance(incoming.get("trajectory"), str):
            merged["trajectory"] = incoming.get("trajectory")
    else:
        if isinstance(base.get("mastery_level"), str):
            merged["mastery_level"] = base.get("mastery_level")
        if isinstance(base.get("trajectory"), str):
            merged["trajectory"] = base.get("trajectory")

    return merged


def _merge_topic_concepts(raw_concepts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    if not isinstance(raw_concepts, dict):
        return ({}, {})

    merged_concepts: dict[str, dict[str, Any]] = {}
    key_to_id: dict[str, str] = {}
    concept_id_map: dict[str, str] = {}

    for raw_id, raw_meta in raw_concepts.items():
        source_id = str(raw_id or "").strip()
        if not source_id:
            continue
        incoming_meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        incoming_label = str(incoming_meta.get("label") or "").strip()
        incoming_id = _to_concept_id(source_id or incoming_label)
        norm_key = _concept_norm_key(incoming_label or incoming_id)

        canonical_id = key_to_id.get(norm_key) if norm_key else ""
        if not canonical_id:
            canonical_id = incoming_id
        if canonical_id in merged_concepts:
            preferred = _prefer_concept_id(canonical_id, incoming_id)
            if preferred != canonical_id:
                merged_concepts[preferred] = merged_concepts.pop(canonical_id)
                for key, value in list(key_to_id.items()):
                    if value == canonical_id:
                        key_to_id[key] = preferred
                for key, value in list(concept_id_map.items()):
                    if value == canonical_id:
                        concept_id_map[key] = preferred
                canonical_id = preferred

        existing_meta = merged_concepts.get(canonical_id, {})
        merged_meta = _merge_concept_meta(existing_meta, incoming_meta, canonical_id)
        merged_concepts[canonical_id] = merged_meta

        if norm_key:
            key_to_id[norm_key] = canonical_id
        concept_id_map[source_id] = canonical_id
        concept_id_map[_to_concept_id(source_id)] = canonical_id
        if incoming_label:
            concept_id_map[_to_concept_id(incoming_label)] = canonical_id

    for canonical_id in list(merged_concepts.keys()):
        concept_id_map[canonical_id] = canonical_id

    return (merged_concepts, concept_id_map)


def _canonical_concept_for_model(
    *,
    topic: str,
    concept_id: str,
    topics: dict[str, Any],
    concept_maps: dict[str, dict[str, str]],
) -> str:
    source_id = str(concept_id or "").strip()
    if not source_id:
        return "concept"

    topic_name = _canonical_topic_for_model(topic, topics)
    topic_map = concept_maps.get(topic_name, {})
    direct = topic_map.get(source_id)
    if isinstance(direct, str) and direct.strip():
        return direct

    source_norm = _to_concept_id(source_id)
    mapped = topic_map.get(source_norm)
    if isinstance(mapped, str) and mapped.strip():
        return mapped

    topic_data = topics.get(topic_name)
    if isinstance(topic_data, dict):
        concepts = topic_data.get("concepts")
        if isinstance(concepts, dict):
            if source_id in concepts:
                return source_id
            if source_norm in concepts:
                return source_norm
    return source_norm or "concept"


def default_student_model() -> dict[str, Any]:
    """Create a fresh multi-topic student model."""
    return {
        "topics": {},  # Structure: {topic_name: {concepts: {}, last_studied: str, total_practice: int}}
        "review_queue": [],
        "active_review": {},
        "paused_reviews": [],
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
    merged_topics: dict[str, Any] = {}
    for raw_topic, raw_topic_data in topics.items():
        if not isinstance(raw_topic_data, dict):
            continue
        topic_name = _canonical_topic_for_model(str(raw_topic), merged_topics)
        incoming = dict(raw_topic_data)
        incoming_concepts = incoming.get("concepts")
        if not isinstance(incoming_concepts, dict):
            incoming_concepts = {}
        incoming["concepts"] = incoming_concepts

        if topic_name not in merged_topics:
            merged_topics[topic_name] = incoming
            continue

        existing = merged_topics.get(topic_name)
        if not isinstance(existing, dict):
            existing = {"concepts": {}, "last_studied": None, "total_practice": 0}

        existing_concepts = existing.get("concepts")
        if not isinstance(existing_concepts, dict):
            existing_concepts = {}
        for concept_id, meta in incoming_concepts.items():
            if not isinstance(concept_id, str) or not isinstance(meta, dict):
                continue
            if concept_id not in existing_concepts:
                existing_concepts[concept_id] = meta
        existing["concepts"] = existing_concepts

        existing_studied = _parse_iso(existing.get("last_studied"))
        incoming_studied = _parse_iso(incoming.get("last_studied"))
        if incoming_studied and (existing_studied is None or incoming_studied > existing_studied):
            existing["last_studied"] = incoming_studied.isoformat(timespec="seconds")

        try:
            existing_total = int(existing.get("total_practice", 0))
        except (TypeError, ValueError):
            existing_total = 0
        try:
            incoming_total = int(incoming.get("total_practice", 0))
        except (TypeError, ValueError):
            incoming_total = 0
        existing["total_practice"] = max(0, existing_total + incoming_total)
        merged_topics[topic_name] = existing

    concept_maps_by_topic: dict[str, dict[str, str]] = {}
    for topic_name, topic_data in list(merged_topics.items()):
        if not isinstance(topic_data, dict):
            continue
        concepts = topic_data.get("concepts")
        merged_concepts, concept_map = _merge_topic_concepts(concepts if isinstance(concepts, dict) else {})
        topic_data["concepts"] = merged_concepts
        merged_topics[topic_name] = topic_data
        concept_maps_by_topic[topic_name] = concept_map

    safe["topics"] = merged_topics

    turn_log = safe.get("turn_log")
    if not isinstance(turn_log, list):
        turn_log = []
    safe["turn_log"] = turn_log

    review_queue = safe.get("review_queue")
    if not isinstance(review_queue, list):
        review_queue = []
    normalized_queue: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in review_queue:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        topic_name = _canonical_topic_for_model(str(row.get("topic") or "General"), safe["topics"])
        concept_id = _canonical_concept_for_model(
            topic=topic_name,
            concept_id=str(row.get("concept_id") or ""),
            topics=safe["topics"],
            concept_maps=concept_maps_by_topic,
        )
        row["topic"] = topic_name
        row["concept_id"] = concept_id
        topic_data = safe["topics"].get(topic_name)
        if isinstance(topic_data, dict):
            concepts = topic_data.get("concepts")
            if isinstance(concepts, dict):
                concept_meta = concepts.get(concept_id)
                if isinstance(concept_meta, dict):
                    canonical_label = str(concept_meta.get("label") or "").strip()
                    if canonical_label:
                        row["label"] = canonical_label
        key = (topic_name, concept_id)
        existing = normalized_queue.get(key)
        if not isinstance(existing, dict):
            normalized_queue[key] = row
            continue

        existing_reviewed = _parse_iso(existing.get("last_reviewed"))
        row_reviewed = _parse_iso(row.get("last_reviewed"))
        if row_reviewed and (existing_reviewed is None or row_reviewed >= existing_reviewed):
            normalized_queue[key] = row
    safe["review_queue"] = list(normalized_queue.values())

    active_review = safe.get("active_review")
    if not isinstance(active_review, dict):
        active_review = {}
    if active_review:
        topic_name = _canonical_topic_for_model(str(active_review.get("topic") or "General"), safe["topics"])
        concept_id = _canonical_concept_for_model(
            topic=topic_name,
            concept_id=str(active_review.get("concept_id") or ""),
            topics=safe["topics"],
            concept_maps=concept_maps_by_topic,
        )
        active_review["topic"] = topic_name
        active_review["concept_id"] = concept_id
        topic_data = safe["topics"].get(topic_name)
        if isinstance(topic_data, dict):
            concepts = topic_data.get("concepts")
            if isinstance(concepts, dict):
                concept_meta = concepts.get(concept_id)
                if isinstance(concept_meta, dict):
                    canonical_label = str(concept_meta.get("label") or "").strip()
                    if canonical_label:
                        active_review["label"] = canonical_label
        status = active_review.get("status")
        if status not in {"awaiting_answer", "paused", "completed"}:
            awaiting = bool(active_review.get("awaiting_answer"))
            status = "awaiting_answer" if awaiting else "completed"
        active_review["status"] = status
        active_review["awaiting_answer"] = status == "awaiting_answer"
    safe["active_review"] = active_review

    paused_reviews = safe.get("paused_reviews")
    if not isinstance(paused_reviews, list):
        paused_reviews = []
    normalized_paused: list[dict[str, Any]] = []
    for item in paused_reviews:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        topic_name = _canonical_topic_for_model(str(row.get("topic") or "General"), safe["topics"])
        concept_id = _canonical_concept_for_model(
            topic=topic_name,
            concept_id=str(row.get("concept_id") or ""),
            topics=safe["topics"],
            concept_maps=concept_maps_by_topic,
        )
        row["topic"] = topic_name
        row["concept_id"] = concept_id
        topic_data = safe["topics"].get(topic_name)
        if isinstance(topic_data, dict):
            concepts = topic_data.get("concepts")
            if isinstance(concepts, dict):
                concept_meta = concepts.get(concept_id)
                if isinstance(concept_meta, dict):
                    canonical_label = str(concept_meta.get("label") or "").strip()
                    if canonical_label:
                        row["label"] = canonical_label
        row["status"] = "paused"
        row["awaiting_answer"] = False
        normalized_paused.append(row)
    paused_reviews = normalized_paused[-8:]
    safe["paused_reviews"] = paused_reviews

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


def migrate_all_student_models() -> dict[str, int]:
    """One-time normalization pass for all persisted students."""
    all_data = _read_all()
    if not isinstance(all_data, dict):
        return {"total_students": 0, "updated_students": 0}

    normalized: dict[str, Any] = {}
    total_students = 0
    updated_students = 0
    for student_id, raw in all_data.items():
        total_students += 1
        source = raw if isinstance(raw, dict) else default_student_model()
        normalized_model = ensure_model_shape(source)
        normalized[student_id] = normalized_model
        if normalized_model != source:
            updated_students += 1

    if updated_students > 0:
        _write_all(normalized)

    return {"total_students": total_students, "updated_students": updated_students}


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
    topic_scope = _canonical_topic_for_model(current_topic, safe.get("topics", {})) if current_topic else ""
    topic_filter = topic_scope.lower() if topic_scope else ""

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

    topic_norm = _canonical_topic_for_model(topic or "General", safe.get("topics", {}))
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
    topic_scope = _canonical_topic_for_model(current_topic, safe.get("topics", {})) if current_topic else ""
    topic_filter = topic_scope.lower() if topic_scope else ""
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
        scope = topic_scope if topic_scope and topic_scope != "General" else "all topics"
        return f"Review queue ({scope}): {due_count}/{total} due now. No due cards in top {limit}."

    lines = []
    for item in due:
        label = item.get("label") or item.get("concept_id") or "concept"
        topic = item.get("topic") or "General"
        days = item.get("interval_days", 1)
        lines.append(f"{label} [{topic}] (interval {days}d)")

    scope = topic_scope if topic_scope and topic_scope != "General" else "all topics"
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

    safe_model = ensure_model_shape(model)
    topics = safe_model.get("topics", {})
    scoped_topic = _canonical_topic_for_model(current_topic, topics) if current_topic else ""
    
    # If specific topic requested and exists
    if scoped_topic and scoped_topic != "General" and scoped_topic in topics:
        topic_data = topics.get(scoped_topic, {})
        concepts = topic_data.get("concepts", {})
        
        if not isinstance(concepts, dict) or not concepts:
            return f"Topic '{scoped_topic}': No concepts learned yet"
        
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
        
        result = f"Topic: {scoped_topic}\n" + "\n".join(parts) if parts else f"Topic: {scoped_topic} (no concepts yet)"
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
