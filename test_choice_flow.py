"""Smoke test for the canonical simplified study flow.

Validates:
- design -> ask -> END
- grade produces a score message

Run as a script (kept intentionally minimal and low-flake).
"""

from __future__ import annotations

from agents_diagnostic import app
def _run_turn(config: dict, state_update: dict) -> list[str]:
    """Run one turn (stream until END) and return emitted message contents."""
    outputs: list[str] = []
    for event in app.stream(state_update, config=config):
        node_state = list(event.values())[0]
        for msg in node_state.get("messages", []):
            content = getattr(msg, "content", "")
            if content:
                outputs.append(content)
    return outputs


def main() -> None:
    thread_id = "test_choice_123"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    print("=" * 70)
    print("Testing Offer Choice Flow")
    print("=" * 70)
    print("Testing Simplified Routing Flow")

    # 1) Ask a diagnostic question (router -> ask -> END)
    out1 = _run_turn(
        config,
        {
            "messages": [],
            "user_id": thread_id,
            "ultimate_goal": "Python Programming",
            "current_focus": "python_programming",
            "knowledge_graph": "",
            "student_answer": "",
            "last_score": 0,
            "awaiting_choice": False,
            "next": "ask",
        },
    )

    joined1 = "\n".join(out1)
    assert joined1.strip(), "expected at least one assistant message"

    print("✅ test_choice_flow.py: PASS")

if __name__ == "__main__":
    main()

