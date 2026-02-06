"""Inspect the student model stored for a given thread_id.

Usage:
  python inspect_student_model.py <thread_id>

This reads from the external JSON store (student_store.json) and displays:
- Concepts learned with proficiency
- Evidence trail per concept
- Recent turn history
- Last tutor hint
"""

from __future__ import annotations

import sys

from student_store import _read_all


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_student_model.py <thread_id>")
        print("\nAvailable thread IDs:")
        all_data = _read_all()
        if all_data:
            for tid in sorted(all_data.keys()):
                model = all_data[tid]
                topics = model.get("topics", {})
                n_topics = len(topics)
                total_concepts = sum(len(t.get("concepts", {})) for t in topics.values())
                n_turns = len(model.get("turn_log", []))
                print(f"  {tid:40} topics={n_topics} concepts={total_concepts} turns={n_turns}")
        else:
            print("  (none yet)")
        sys.exit(1)

    thread_id = sys.argv[1]
    all_data = _read_all()
    model = all_data.get(thread_id)

    if not model:
        print(f"❌ No student model found for thread_id: {thread_id}")
        return

    print("=" * 80)
    print(f"Student Model: {thread_id}")
    print("=" * 80)
    print(f"Updated:    {model.get('updated_at')}")
    print()

    topics = model.get("topics", {})
    if not topics:
        print("📚 Topics: (none yet)")
    else:
        print(f"📚 Topics ({len(topics)}):")
        for topic_name, topic_data in sorted(topics.items()):
            concepts = topic_data.get("concepts", {})
            last_studied = topic_data.get("last_studied", "never")
            total_practice = topic_data.get("total_practice", 0)
            
            print(f"\n  � {topic_name}")
            print(f"     Last studied: {last_studied[:19] if last_studied != 'never' else 'never'}")
            print(f"     Total practice: {total_practice} updates")
            print(f"     Concepts: {len(concepts)}")
            
            if concepts:
                items = []
                for cid, meta in concepts.items():
                    prof = meta.get("proficiency", 0.0)
                    items.append((prof, cid, meta))
                items.sort(reverse=True)

                for prof, cid, meta in items:
                    label = meta.get("label", cid)
                    aliases = meta.get("aliases", [])
                    notes = meta.get("notes", "")
                    evidence = meta.get("evidence", [])
                    mastery = meta.get("mastery_level", "unknown")
                    trajectory = meta.get("trajectory", "unknown")
                    practice_count = meta.get("practice_count", 0)
                    first_seen = meta.get("first_seen", "?")
                    last_practiced = meta.get("last_practiced", "?")

                    print(f"\n     [{int(prof * 100):3d}%] {label} ({mastery}, {trajectory})")
                    print(f"            ID: {cid}")
                    if aliases:
                        print(f"            Aliases: {', '.join(aliases)}")
                    print(f"            Practice: {practice_count} times | First: {first_seen[:10]} | Last: {last_practiced[:10]}")
                    if notes:
                        print(f"            Notes: {notes}")
                    if evidence:
                        print(f"            Evidence ({len(evidence)} events, showing last 3):")
                        for i, ev in enumerate(evidence[-3:], 1):  # Show last 3
                            kind = ev.get("kind", "?")
                            text = ev.get("text", "")
                            delta = ev.get("delta", 0.0)
                            prof_before = ev.get("proficiency_before", 0.0)
                            prof_after = ev.get("proficiency_after", 0.0)
                            reasoning = ev.get("reasoning", "")
                            print(f"              {i}. [{kind:20}] Δ{delta:+.2f} ({prof_before:.2f}→{prof_after:.2f})")
                            print(f"                 {text[:70]}")
                            if reasoning:
                                print(f"                 Why: {reasoning[:60]}")

    hint = model.get("last_tutor_hint")
    if hint:
        print(f"\n💡 Last Tutor Hint:\n   {hint}")

    turn_log = model.get("turn_log", [])
    if turn_log:
        print(f"\n💬 Turn History (last 3 of {len(turn_log)}):")
        for i, turn in enumerate(turn_log[-3:], len(turn_log) - 2):
            user = turn.get("user", "")[:60]
            asst = turn.get("assistant", "")[:60]
            print(f"\n  Turn {i}:")
            print(f"    User: {user}")
            print(f"    AI:   {asst}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
