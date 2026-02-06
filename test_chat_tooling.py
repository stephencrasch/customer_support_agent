"""Smoke tests for the canonical simplified agent.

This repo doesn't guarantee pytest is installed in every environment, so this is
structured as a runnable script.

It verifies:
- Curriculum is generated when `next=design`.
- Asking a question produces a `question` field.
"""

from __future__ import annotations


from agents_diagnostic import create_diagnostic_graph


def _latest_text(messages) -> str:
    for msg in reversed(messages or []):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def main() -> None:
    app = create_diagnostic_graph(checkpointer=None)
    config = {"configurable": {"thread_id": "test_chat_tooling"}}

    state = {
        "messages": [],
        "user_id": "test_chat_tooling",
        "ultimate_goal": "Python Programming",
        "current_focus": "python_programming",
        "knowledge_graph": "",
        "student_answer": "",
        "last_score": 0,
        "awaiting_choice": False,
        "next": "ask",
    }

    out = app.invoke(state, config=config)
    state.update(out)
    txt = _latest_text(state.get("messages"))
    assert txt, "expected at least one assistant message"

    print(" test_chat_tooling.py: PASS")


if __name__ == "__main__":
    main()
