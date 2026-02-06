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
from datetime import datetime, timezone
from typing import Any


_STORE_PATH = os.path.join(os.path.dirname(__file__), "student_store.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        "turn_log": [],
        "updated_at": _utc_now_iso(),
    }


def load_student_model(*, student_id: str) -> dict[str, Any]:
    """Load a student's model from disk, or create one if missing."""
    all_data = _read_all()
    model = all_data.get(student_id)
    if not isinstance(model, dict):
        model = default_student_model()
    # Ensure topics structure exists
    if "topics" not in model:
        model["topics"] = {}
    return model


def save_student_model(*, student_id: str, model: dict[str, Any]) -> None:
    """Persist a student's model to disk."""
    all_data = _read_all()
    model = dict(model)
    model["updated_at"] = _utc_now_iso()
    all_data[student_id] = model
    _write_all(all_data)


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

