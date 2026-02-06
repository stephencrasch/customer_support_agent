"""Tools used by the diagnostic tutor (tool-calling).

Kept separate from the flower shop/customer support tools in `tools.py`.
"""

from __future__ import annotations

import json
from json import JSONDecodeError

from langchain_core.tools import tool

from knowledge_graph import KnowledgeGraph, KnowledgeNode


def _to_snake_case(text: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _safe_concept_id(label: str | None, ultimate_goal: str) -> str:
    if not label:
        return _to_snake_case(ultimate_goal)
    return _to_snake_case(label)


@tool
def query_learning_progress(knowledge_graph_json: str, query: str = "summary") -> str:
    """Query the student's learning progress.

    This is intentionally narrow and deterministic so the chat node can rely on it.

    Args:
      knowledge_graph_json: The serialized KnowledgeGraph JSON (from state).
      query: One of:
        - "summary": concise, human-readable progress snapshot
        - "graph_json": dump the raw knowledge graph JSON for inspection
    """

    k = (query or "summary").strip().lower()
    if not knowledge_graph_json:
        return "No knowledge graph exists yet. Ask/answer a graded question first so I can start tracking concepts."

    if k in {"graph", "graph_json", "json", "dump", "raw"}:
        return knowledge_graph_json

    try:
        graph = KnowledgeGraph.from_json(knowledge_graph_json)
    except (TypeError, ValueError, JSONDecodeError):
        return "I couldn't parse the saved knowledge graph JSON."

    if not graph.nodes:
        return "Your knowledge graph is empty so far. Answer a graded question first so I can start tracking concepts."

    concepts = list(graph.nodes.values())
    mastered = [n for n in concepts if n.proficiency >= 80]
    learning = [n for n in concepts if 40 <= n.proficiency < 80]
    struggling = [n for n in concepts if n.proficiency < 40]

    def _fmt(nodes: list[KnowledgeNode]) -> str:
        if not nodes:
            return "None yet"
        return ", ".join(
            [f"{n.name} ({n.proficiency:.0f}%)" for n in sorted(nodes, key=lambda x: -x.proficiency)]
        )

    overall = sum(n.proficiency for n in concepts) / len(concepts)
    return (
        "Progress snapshot:\n"
        f"- Concepts tracked: {len(concepts)}\n"
        f"- Mastered ({len(mastered)}): {_fmt(mastered)}\n"
        f"- Learning ({len(learning)}): {_fmt(learning)}\n"
        f"- Needs work ({len(struggling)}): {_fmt(struggling)}\n"
        f"- Overall average: {overall:.0f}%"
    )


@tool
def canonicalize_concepts(
    *,
    ultimate_goal: str,
    existing_concept_ids_json: str,
    raw_concepts_json: str,
) -> str:
    """Canonicalize raw concept labels into stable concept ids.

    This tool is designed for LLM tool-calling: it always returns a JSON string.

    Inputs are JSON strings (not Python lists) because tool-call payloads are JSON.

    Returns JSON:
      {
        "canonical": {"raw label": "canonical_id", ...},
        "reason": "..."
      }
    """

    # Parse inputs.
    try:
        existing_concept_ids = json.loads(existing_concept_ids_json or "[]")
        if not isinstance(existing_concept_ids, list):
            existing_concept_ids = []
    except (TypeError, ValueError, JSONDecodeError):
        existing_concept_ids = []

    try:
        raw_concepts = json.loads(raw_concepts_json or "[]")
        if not isinstance(raw_concepts, list):
            raw_concepts = []
    except (TypeError, ValueError, JSONDecodeError):
        raw_concepts = []

    cleaned: list[str] = []
    for c in raw_concepts:
        if isinstance(c, str) and c.strip():
            cleaned.append(c.strip())
    cleaned = list(dict.fromkeys(cleaned))

    if not cleaned:
        return json.dumps({"canonical": {}, "reason": "no concepts"}, ensure_ascii=False)

    # Deterministic, always-available fallback behavior:
    # - if exact match to an existing id, keep it
    # - otherwise snake_case the string
    existing = {str(x) for x in existing_concept_ids if isinstance(x, str) and str(x).strip()}

    canonical: dict[str, str] = {}
    for raw in cleaned:
        raw_snake = _to_snake_case(raw)
        if raw in existing:
            canonical[raw] = raw
        elif raw_snake in existing:
            canonical[raw] = raw_snake
        else:
            canonical[raw] = _safe_concept_id(raw, ultimate_goal)

    return json.dumps({"canonical": canonical, "reason": "deterministic"}, ensure_ascii=False)
