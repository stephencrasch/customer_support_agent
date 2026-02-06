"""Interactive chat for the diagnostic-driven adaptive learning system.

This CLI drives `agents_diagnostic.app` turn-by-turn.
"""

from __future__ import annotations

import uuid

from agents import StudyState, app


def _print_new_messages(messages: list, *, start_index: int) -> int:
    """Print messages[start_index:] and return the new cursor."""

    if not messages:
        return start_index

    for msg in messages[start_index:]:
        content = getattr(msg, "content", "")
        if content:
            print(f"\n🤖 AI: {content}")

    return len(messages)


def _merge_and_print(
    *,
    state: StudyState,
    node_state: dict,
    printed_upto: int,
) -> int:
    """Merge streamed node_state into our local state and print only truly-new messages.

    LangGraph streams node-local state snapshots. Some nodes emit the full
    message history, which can cause re-printing if we naively print whatever
    is present in each event.

    Contract here:
    - keep `state["messages"]` as the full history
    - only print delta messages that appear after `printed_upto`
    """

    incoming = node_state.get("messages")
    if not isinstance(incoming, list):
        return printed_upto

    # Prefer the longer history as the source of truth.
    if len(incoming) >= len(state.get("messages", [])):
        state["messages"] = incoming

    return _print_new_messages(state["messages"], start_index=printed_upto)


def chat_session() -> None:
    print("=" * 70)
    print("🎓 DIAGNOSTIC ADAPTIVE LEARNING - INTERACTIVE CHAT")
    print("=" * 70)

    goal = input("\n📚 What would you like to learn about? ").strip()
    if not goal:
        print("\n❌ Please enter a learning goal.")
        return

    user_id = f"user_{uuid.uuid4().hex[:8]}"

    print("\n✅ Starting session")
    print(f"   Goal: {goal}")
    print(f"   User ID: {user_id}")
    print("-" * 70)

    state: StudyState = {
        "messages": [],
        "user_id": user_id,
        "knowledge_graph": "",
        "ultimate_goal": goal,
        "current_focus": goal.lower().replace(" ", "_"),
        "student_answer": "",
        "last_score": 0,
        "awaiting_choice": False,
        "last_tutor_message": "",
        "last_dialogue_update_user_text": "",
        "next": "ask",
    }

    printed_upto = 0

    config = {"configurable": {"thread_id": user_id}, "recursion_limit": 50}

    # Kick off: router -> ask -> END
    for event in app.stream(state, config=config):
        node_state = list(event.values())[0]
        printed_upto = _merge_and_print(state=state, node_state=node_state, printed_upto=printed_upto)

    while True:
        current_state = app.get_state(config).values

        # NOTE: In this project, `next == "end"` is used as a *stop marker*
        # so the graph emits at most one assistant message per run.
        # We should only terminate the CLI session when the assistant explicitly
        # produced the end-session message.
        if current_state.get("next") == "end":
            # Persisted state contains full message history.
            state["messages"] = current_state.get("messages", [])
            printed_upto = _print_new_messages(state["messages"], start_index=printed_upto)

            last_text = ""
            if state.get("messages"):
                last_text = (getattr(state["messages"][-1], "content", "") or "").strip()

            if last_text.startswith("🎉 Congratulations!"):
                break

            # Normal case: after one assistant message, we prompt the user.

        print("\n" + "-" * 70)
        user_text = input("\n👤 You: ").strip()

        if user_text.lower() in {"quit", "exit", "stop"}:
            print("\n👋 Session paused. Your progress is saved!")
            print(f"   Your user ID: {user_id}")
            break

        if not user_text:
            print("⚠️  Please provide an answer.")
            continue

        updated_state = {**current_state, "student_answer": user_text}

        for event in app.stream(updated_state, config=config):
            node_state = list(event.values())[0]
            printed_upto = _merge_and_print(state=state, node_state=node_state, printed_upto=printed_upto)


def main() -> None:
    print("\n" + "=" * 70)
    print("🎓 DIAGNOSTIC ADAPTIVE LEARNING SYSTEM")
    print("=" * 70)
    print("\nOptions:")
    print("  1. Start new learning session")
    print("  2. Exit")

    choice = input("\nYour choice (1-2): ").strip()

    if choice == "1":
        chat_session()
    elif choice == "2":
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted. Your progress is saved!")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
