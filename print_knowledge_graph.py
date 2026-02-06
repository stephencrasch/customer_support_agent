"""Print the current knowledge graph for a given LangGraph thread.

This is a small debugging/inspection utility.

Usage:
  python print_knowledge_graph.py <thread_id>

Notes:
- This uses the checkpointer in `agents_simplified.py`.
  for the current Python process that created the state.
- If you want persistence across processes, we can switch back to a SQLite
  checkpointer once the repo pins a langgraph version that ships one, or we can
  implement a tiny local SQLite saver.
"""

from __future__ import annotations

import sys

from agents_diagnostic import app
from knowledge_graph import KnowledgeGraph


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python print_knowledge_graph.py <thread_id>")

    thread_id = sys.argv[1]
    config = {"configurable": {"thread_id": thread_id}}

    state = app.get_state(config).values
    kg_json = state.get("knowledge_graph")
    if not kg_json:
        print("No knowledge graph found yet. Answer at least one graded question first.")
        return

    kg = KnowledgeGraph.from_json(kg_json)

    print("=" * 80)
    print(f"Knowledge Graph for thread_id={thread_id}")
    print("=" * 80)
    print(f"Goal: {state.get('ultimate_goal')}")
    print(f"Current focus: {state.get('current_focus')}")
    print(f"Concepts tracked: {len(kg.nodes)}")
    print("-")

    nodes = sorted(kg.nodes.values(), key=lambda n: (-float(n.proficiency), n.id))
    for n in nodes:
        print(f"[{n.status.value:11}] {n.name} ({n.id})")
        print(f"  proficiency: {n.proficiency:.0f}%   attempts: {n.attempts}")
        if n.taught_points:
            print("  taught_points:")
            for p in n.taught_points[-5:]:
                print(f"    - {p}")
        if n.misconceptions:
            print("  misconceptions:")
            for m in n.misconceptions[-3:]:
                print(f"    - {m}")
        if n.last_check_question:
            print(f"  last_check_question: {n.last_check_question}")
        if n.prerequisites:
            print("  prerequisites:", ", ".join(n.prerequisites))
        print("-")


if __name__ == "__main__":
    main()
