"""List/print saved knowledge graphs.

This reads files saved by `graph_persistence.save_knowledge_graph()`.

Usage:
  python view_saved_graphs.py list
  python view_saved_graphs.py print <thread_id>

Examples:
  python view_saved_graphs.py list
  python view_saved_graphs.py print user_1234abcd
"""

from __future__ import annotations

import sys

from graph_persistence import list_saved_thread_ids, load_knowledge_graph


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python view_saved_graphs.py list | print <thread_id>")

    cmd = sys.argv[1].lower()

    if cmd == "list":
        ids = list_saved_thread_ids()
        if not ids:
            print("No saved knowledge graphs found yet.")
            return
        print("Saved knowledge graphs:")
        for tid in ids:
            print("-", tid)
        return

    if cmd == "print":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: python view_saved_graphs.py print <thread_id>")
        tid = sys.argv[2]
        data = load_knowledge_graph(thread_id=tid)
        if not data:
            print(f"No saved knowledge graph for thread_id={tid}")
            return
        print(data)
        return

    raise SystemExit("Unknown command. Use: list | print <thread_id>")


if __name__ == "__main__":
    main()
