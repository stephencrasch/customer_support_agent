"""Interactive CLI for `tutor_agent.app`.

Goal: a tiny, reliable loop for testing LangGraph turn-by-turn.

How it works:
- We keep a fixed `thread_id` so the MemorySaver checkpointer can persist state
  across turns.
- Each user input is passed in as `student_answer`.
- The graph runs router -> chat -> END.
- We print only the *new* AI messages since the last turn.

Exit commands:
- quit / exit
"""

from __future__ import annotations

import uuid

from tutor_agent import app


def _new_ai_messages(state: dict, *, cursor: int) -> tuple[list[str], int]:
    """Return (new_ai_texts, new_cursor) from state['messages'].

    We track a cursor so we don't re-print past messages.
    """

    messages = state.get("messages") or []
    new = messages[cursor:]

    out: list[str] = []
    for m in new:
        if getattr(m, "type", None) == "ai":
            text = (getattr(m, "content", "") or "").strip()
            if text:
                out.append(text)

    return out, len(messages)


def main() -> None:
    print("=" * 70)
    print("🎓 Tutor Agent (LangGraph) - Interactive Chat")
    print("=" * 70)

    # Thread id is what the checkpointer uses to load/save state.
    thread_id = f"tutor_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print("\n✅ Session started")
    print(f"   Thread: {thread_id}")
    print("\n💡 Just start chatting - I'll figure out what you want to learn!")

    # First run: empty student_answer to trigger a greeting.
    state = app.invoke({"current_topic": "", "student_answer": "", "messages": []}, config=config)
    cursor = 0
    ai_texts, cursor = _new_ai_messages(state, cursor=cursor)
    for t in ai_texts:
        print(f"\n🤖 AI: {t}")

    while True:
        print("\n" + "-" * 70)
        user_text = input("\n👤 You: ").strip()

        if user_text.lower() in {"quit", "exit"}:
            print("\n👋 Bye!")
            break

        if not user_text:
            print("⚠️  Please type something (or 'quit').")
            continue

        state = app.invoke({"student_answer": user_text}, config=config)
        ai_texts, cursor = _new_ai_messages(state, cursor=cursor)

        if not ai_texts:
            print("\n🤖 AI: (no response)")
        else:
            for t in ai_texts:
                print(f"\n🤖 AI: {t}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Bye!")
