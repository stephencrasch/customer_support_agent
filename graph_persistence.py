"""Lightweight persistence for the diagnostic tutor's knowledge graph.

This intentionally does NOT try to persist the full LangGraph state/checkpoints.
It only saves the `knowledge_graph` JSON out to disk so you can inspect progress
across program restarts.

Why this exists:
- This repo currently pins `langgraph==1.0.1` which doesn't ship a sqlite
  checkpointer module.
- `MemorySaver()` is process-local.

So we persist the one thing people most want: the knowledge graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


_DEFAULT_DIR = Path(".knowledge_graphs")


def _safe_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in name)


def save_knowledge_graph(*, thread_id: str, knowledge_graph_json: str, base_dir: Path | None = None) -> Path:
    """Save the knowledge graph JSON for a thread_id to disk."""
    base = base_dir or _DEFAULT_DIR
    base.mkdir(parents=True, exist_ok=True)

    safe = _safe_filename(thread_id)
    path = base / f"{safe}.json"

    # Pretty-print if valid JSON; otherwise save raw string.
    try:
        obj = json.loads(knowledge_graph_json)
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    except (ValueError, TypeError, OSError):
        path.write_text(knowledge_graph_json or "", encoding="utf-8")

    return path


def load_knowledge_graph(*, thread_id: str, base_dir: Path | None = None) -> Optional[str]:
    base = base_dir or _DEFAULT_DIR
    safe = _safe_filename(thread_id)
    path = base / f"{safe}.json"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def list_saved_thread_ids(*, base_dir: Path | None = None) -> list[str]:
    base = base_dir or _DEFAULT_DIR
    if not base.exists():
        return []
    return sorted([p.stem for p in base.glob("*.json")])
