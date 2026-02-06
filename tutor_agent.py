"""
Diagnostic-Driven Adaptive Learning System
Starting simple: router → chat flow

This file is intentionally minimal so you can learn LangGraph basics:
- Define a typed state (TutorState)
- Define node functions that read/return partial state updates
- Build a StateGraph with nodes + edges
- Compile to an app you can invoke repeatedly (one user turn at a time)

Key idea:
Each call to `app.invoke(...)` represents ONE "turn" of work.
We end the graph run at END after the assistant produces a message,
so the outer UI/CLI can collect the next user input and call `app.invoke` again.
"""

from __future__ import annotations

import json
from typing import Annotated, Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from student_store import (
  load_student_model as store_load_student_model,
  save_student_model as store_save_student_model,
  model_snapshot,
)

# Loads environment variables from .env (e.g., OPENAI_API_KEY)
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

# Your model "engine". All LLM calls in nodes will use this.
# Tip: keep temperature lower for more deterministic behavior while debugging.
# Some models (including certain "gpt-5" endpoints) may not support arbitrary
# temperature values. For a smooth learning experience, we use a widely
# compatible model + temperature.
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ============================================================================
# State Definition (MINIMAL)
# ============================================================================

class TutorState(TypedDict):
    """Shared state that flows through the graph.

    LangGraph passes a *single* state object through nodes.
    Each node returns a partial update dict; LangGraph merges it into the state.

    Fields:
      - messages: conversation history (managed via `add_messages` reducer)
      - current_topic: what topic the student is currently learning (inferred from conversation)
      - student_answer: latest user input for this turn (cleared after use)
      - student_model: external JSON-backed student model keyed by thread_id

    Note:
      - The `messages` field uses a reducer (`add_messages`) so that when a node
        returns {"messages": [AIMessage(...)]}, LangGraph appends it to history.
      - For non-reducer fields (like current_topic), a node can overwrite them.
    """

    # The Annotated + add_messages reducer is the idiomatic LangGraph pattern
    # for accumulating message history.
    messages: Annotated[list[BaseMessage], add_messages]

    # Current topic being discussed (inferred by watcher, can change mid-session)
    current_topic: str

    # Latest user input for this turn.
    student_answer: str

    # Externalized student model (loaded/saved using thread_id).
    # Keeping it in state makes it easy for nodes to read without tool-calling.
    student_model: Dict[str, Any]

    # A short internal hint from the watcher to guide the tutor's next reply.
    # This should NOT be shown verbatim to the student unless the tutor chooses to.
    tutor_hint: str

    # Pending watcher outputs for this turn (merged into student_model later).
    # Kept transient so we can merge deterministically.
    concept_updates: List[Dict[str, Any]]

# ============================================================================
# Helper Functions
# ============================================================================

def extract_json(content: str) -> dict:
    """Extract JSON from an LLM response (handles markdown fences)."""

    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _collect_concept_labels(model: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return mapping: concept_id -> list of names (label + aliases) normalized."""

    concepts = model.get("concepts")
    if not isinstance(concepts, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for cid, meta in concepts.items():
        if not isinstance(cid, str) or not isinstance(meta, dict):
            continue
        names: List[str] = []
        label = meta.get("label")
        if isinstance(label, str) and label.strip():
            names.append(_norm_text(label))
        aliases = meta.get("aliases")
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    names.append(_norm_text(a))
        # Always include id itself for matching.
        names.append(_norm_text(cid))
        # De-dupe while preserving order.
        seen = set()
        uniq: List[str] = []
        for n in names:
            if n and n not in seen:
                seen.add(n)
                uniq.append(n)
        out[cid] = uniq

    return out

def _concepts_mentioned_in_text(model: Dict[str, Any], text: str) -> List[str]:
    """Find existing concept_ids whose label/aliases appear in `text` (substring match)."""

    t = _norm_text(text)
    if not t:
        return []

    labels = _collect_concept_labels(model)
    mentioned: List[str] = []
    for cid, names in labels.items():
        if any(n and n in t for n in names):
            mentioned.append(cid)
    return mentioned


def _to_concept_id(s: str) -> str:
    """Best-effort stable id generator (snake-ish)."""

    s = (s or "").strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    cid = "".join(out).strip("_")
    return cid or "concept"


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        v = 0.0
    return max(0.0, min(1.0, v))


def _merge_concept_updates(
    *,
    model: Dict[str, Any],
    updates: List[Dict[str, Any]],
    current_topic: str,
    evidence_per_concept: int = 12,
) -> Dict[str, Any]:
    """Deterministically merge watcher updates into the stored student model.

    Rules:
    - Concepts are scoped to current_topic
    - Concept dedupe: map incoming updates to existing concepts within topic if label/alias matches.
    - Proficiency updated via delta (clamped to [0,1]).
    - Mastery level, trajectory, temporal fields tracked.
    - Evidence appended with timestamp and proficiency delta; bounded to last N per concept.
    """

    from datetime import datetime, timezone

    model = dict(model or {})
    
    # Navigate to topics structure
    topics = model.get("topics", {})
    if not isinstance(topics, dict):
        topics = {}
    
    topic_data = topics.get(current_topic, {})
    if not isinstance(topic_data, dict):
        topic_data = {"concepts": {}, "last_studied": None, "total_practice": 0}
    
    concepts = topic_data.get("concepts", {})
    if not isinstance(concepts, dict):
        concepts = {}

    # Build label index SCOPED TO THIS TOPIC ONLY
    labels = _collect_concept_labels({"concepts": concepts})
    # Reverse index from normalized name/alias -> canonical concept_id
    name_to_id: Dict[str, str] = {}
    for cid, names in labels.items():
        for n in names:
            if n and n not in name_to_id:
                name_to_id[n] = cid

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for upd in updates or []:
        if not isinstance(upd, dict):
            continue

        label = upd.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        norm_label = _norm_text(label)

        # Identify target concept id.
        requested_id = upd.get("concept_id")
        cid_guess = requested_id if isinstance(requested_id, str) and requested_id.strip() else _to_concept_id(label)
        cid_guess = _to_concept_id(cid_guess)

        target_cid = name_to_id.get(norm_label) or name_to_id.get(_norm_text(cid_guess))
        if not target_cid:
            aliases = upd.get("aliases")
            if isinstance(aliases, list):
                for a in aliases:
                    if isinstance(a, str) and a.strip():
                        target_cid = name_to_id.get(_norm_text(a))
                        if target_cid:
                            break

        if not target_cid:
            target_cid = cid_guess

        meta = concepts.get(target_cid)
        is_new = not isinstance(meta, dict) or not meta
        if not isinstance(meta, dict):
            meta = {}

        # Label + aliases.
        meta.setdefault("label", label.strip())
        if isinstance(meta.get("label"), str) and meta.get("label").strip() != label.strip():
            # Keep existing label; add as alias instead.
            meta_aliases = meta.get("aliases")
            if not isinstance(meta_aliases, list):
                meta_aliases = []
            if label.strip() not in meta_aliases:
                meta_aliases.append(label.strip())
            meta["aliases"] = meta_aliases

        incoming_aliases = upd.get("aliases")
        if isinstance(incoming_aliases, list):
            meta_aliases = meta.get("aliases")
            if not isinstance(meta_aliases, list):
                meta_aliases = []
            for a in incoming_aliases:
                if isinstance(a, str) and a.strip() and a.strip() not in meta_aliases:
                    meta_aliases.append(a.strip())
            meta["aliases"] = meta_aliases

        # Proficiency update via delta.
        old_prof = meta.get("proficiency", 0.0)
        delta = 0.0
        if "proficiency_delta" in upd:
            try:
                delta = float(upd["proficiency_delta"])
            except (TypeError, ValueError):
                delta = 0.0
        
        new_prof = _clamp01(old_prof + delta)
        meta["proficiency"] = new_prof

        # Mastery level.
        mastery = upd.get("mastery_level")
        if mastery in ("introduced", "developing", "proficient", "mastered"):
            meta["mastery_level"] = mastery
        elif new_prof < 0.3:
            meta["mastery_level"] = "introduced"
        elif new_prof < 0.6:
            meta["mastery_level"] = "developing"
        elif new_prof < 0.85:
            meta["mastery_level"] = "proficient"
        else:
            meta["mastery_level"] = "mastered"

        # Trajectory (simple heuristic: delta direction).
        if delta > 0.05:
            meta["trajectory"] = "improving"
        elif delta < -0.05:
            meta["trajectory"] = "declining"
        else:
            meta["trajectory"] = "stable"

        # Temporal tracking.
        if is_new:
            meta["first_seen"] = now
        meta["last_practiced"] = now
        practice_count = meta.get("practice_count", 0)
        if not isinstance(practice_count, int):
            practice_count = 0
        meta["practice_count"] = practice_count + 1

        # Notes.
        if isinstance(upd.get("notes"), str) and upd.get("notes").strip():
            meta["notes"] = upd.get("notes").strip()

        # Evidence event with enhanced metadata.
        ev = upd.get("evidence")
        if isinstance(ev, dict):
            evidence = meta.get("evidence")
            if not isinstance(evidence, list):
                evidence = []
            event = {
                "kind": ev.get("kind", "observation"),
                "text": ev.get("text", ""),
                "source": ev.get("source", "turn"),
                "timestamp": now,
                "proficiency_before": old_prof,
                "proficiency_after": new_prof,
                "delta": delta,
            }
            if "reasoning" in ev:
                event["reasoning"] = ev.get("reasoning", "")
            evidence.append(event)
            meta["evidence"] = evidence[-evidence_per_concept:]

        concepts[target_cid] = meta

        # Update reverse index as we add concepts.
        all_names = []
        if isinstance(meta.get("label"), str):
            all_names.append(_norm_text(meta.get("label")))
        if isinstance(meta.get("aliases"), list):
            all_names.extend(_norm_text(a) for a in meta.get("aliases") if isinstance(a, str))
        all_names.append(_norm_text(target_cid))
        for n in all_names:
            if n and n not in name_to_id:
                name_to_id[n] = target_cid

    # Save back to topic
    topic_data["concepts"] = concepts
    topic_data["last_studied"] = now
    topic_data["total_practice"] = topic_data.get("total_practice", 0) + len(updates)
    topics[current_topic] = topic_data
    model["topics"] = topics
    
    return model


WATCHER_PROMPT = """You are an internal learning observer for a tutoring system.

Previous topic: {current_topic}

STEP 1: Determine the topic for this turn.
- If student explicitly mentions a subject ("Let's learn Python", "teach me NLP"), use that
- If continuing previous conversation, keep current topic
- If unclear, infer from context (e.g., "decorators" → Python, "tokenization" → NLP)
- Use broad categories: "Python", "NLP", "Machine Learning", "Web Development", "Math", etc.
- If truly ambiguous or meta-question ("what should I study?"), use "General"

STEP 2: Analyze learning and update concepts WITHIN that topic.

Student last message:
{student_message}

Tutor last reply:
{tutor_reply}

Existing concepts in topic "{current_topic}":
{related_concepts_json}

Constraints:
- Do NOT invent new concepts if an existing concept already covers it (dedupe by label/aliases).
- Only propose concepts that are relevant to the last turn.
- Each concept update MUST include evidence with proficiency_delta.
- Analyze the turn to estimate how much the student's understanding changed.
- Return ONLY valid JSON (no markdown).

PROFICIENCY DELTA GUIDELINES:
- Student explicitly confused/doesn't understand: -0.1 to -0.2
- Student asks clarifying question: +0.05 to +0.1 (engagement)
- Student gives correct explanation/application: +0.15 to +0.3
- Student gives incorrect answer: -0.05 to -0.15
- Tutor introduces new concept: +0.1 to +0.2 (first exposure)
- Tutor re-explains existing concept: +0.05 to +0.1
- No clear signal: 0.0

MASTERY LEVELS:
- introduced (0.0-0.3): Just seen, minimal understanding
- developing (0.3-0.6): Grasps basics, needs practice
- proficient (0.6-0.85): Solid understanding, occasional errors
- mastered (0.85-1.0): Deep understanding, can teach others

Return JSON with this schema:
{{
  "current_topic": "Python",
  "topic_switched": false,
  "tutor_hint": "1-2 sentences to help the tutor respond next turn.",
  "concept_updates": [
    {{
      "concept_id": "stable_id_snake_case",
      "label": "Human readable concept name",
      "aliases": ["optional", "strings"],
      "proficiency_delta": 0.0,
      "mastery_level": "introduced|developing|proficient|mastered",
      "notes": "optional concise notes",
      "evidence": {{
        "kind": "confused|asked|correct_explanation|correct_application|incorrect|introduced|observation",
        "text": "short quote or paraphrase from the student/tutor turn",
        "source": "student|tutor",
        "reasoning": "why this delta was assigned"
      }}
    }}
  ]
}}
"""


def observe_learning(state: TutorState) -> Dict[str, Any]:
    """LLM watcher: proposes concept updates + internal tutor hint.

    IMPORTANT: This node must be silent (no AIMessage appended). It only returns
    structured data to be merged+persisted later.
    """

    model = state.get("student_model") or {}
    current_topic = state.get("current_topic", "")
    
    last_user = ""
    last_ai = ""
    for msg in reversed(state.get("messages", []) or []):
        if not last_ai and isinstance(msg, AIMessage):
            last_ai = msg.content
        if not last_user and isinstance(msg, HumanMessage):
            last_user = msg.content
        if last_user and last_ai:
            break

    # Get concepts for current topic only
    topics = model.get("topics", {})
    topic_concepts = {}
    if current_topic and isinstance(topics, dict) and current_topic in topics:
        topic_data = topics.get(current_topic, {})
        topic_concepts = topic_data.get("concepts", {}) if isinstance(topic_data, dict) else {}
    
    mentioned = _concepts_mentioned_in_text({"concepts": topic_concepts}, (last_user or "") + "\n" + (last_ai or ""))
    related = {cid: topic_concepts.get(cid) for cid in mentioned if cid in topic_concepts}

    prompt = WATCHER_PROMPT.format(
        current_topic=current_topic or "General",
        related_concepts_json=json.dumps(related, ensure_ascii=False, indent=2, sort_keys=True),
        student_message=last_user,
        tutor_reply=last_ai,
    )

    resp = llm.invoke([{"role": "system", "content": prompt}])
    try:
        data = extract_json(resp.content)
    except Exception:
        # Fail closed: don't break the turn if watcher output is malformed.
        data = {"current_topic": current_topic or "General", "topic_switched": False, "tutor_hint": "", "concept_updates": []}

    new_topic = data.get("current_topic", current_topic or "General")
    if not isinstance(new_topic, str) or not new_topic.strip():
        new_topic = "General"
    
    topic_switched = data.get("topic_switched", False)
    
    tutor_hint = data.get("tutor_hint") if isinstance(data, dict) else ""
    if not isinstance(tutor_hint, str):
        tutor_hint = ""
    tutor_hint = tutor_hint.strip()
    if len(tutor_hint) > 500:
        tutor_hint = tutor_hint[:500].rstrip() + "\n"

    concept_updates = []
    raw_updates = data.get("concept_updates") if isinstance(data, dict) else None
    if isinstance(raw_updates, list):
        for u in raw_updates:
            if isinstance(u, dict):
                concept_updates.append(u)

    return {
        "current_topic": new_topic,
        "tutor_hint": tutor_hint,
        "concept_updates": concept_updates,
    }

# ============================================================================
# Router Node (Simple decision)
# ============================================================================

SIMPLE_ROUTER_PROMPT = """You are a routing assistant for an adaptive learning system.

Student's goal: {ultimate_goal}
Student's message: {student_message}

Decide what to do next. Return ONLY JSON:
{{
  "next": "chat"
}}

For now, always return "chat" - we'll expand routing later.
"""

def router(state: TutorState) -> Dict[str, Any]:
    """Router node.

    In a bigger system, router decides which node should run next.
    For example:
      - "chat" for open conversation
      - "ask" to ask a diagnostic question
      - "diagnose" to grade an answer
      - "teach" to explain a concept

    For this starter graph we keep it deterministic.

    IMPORTANT:
    Router returns *partial state* updates. Here it sets a transient key "next"
    used only to choose an outgoing edge. We don't need to add "next" to
    TutorState because it's not part of the long-lived state we care about.
    """

    student_text = (state.get("student_answer") or "").strip()

    # If no input yet (first turn), still load model so chat has snapshot.
    if not student_text:
        return {"next": "load_student_model"}

    # For now, always load model then chat.
    return {"next": "load_student_model"}

# ==========================================================================
# Silent nodes: load/update external student model
# ==========================================================================

def load_student_model(state: TutorState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load (or create) the external student model for this thread."""

    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id") or "default"

    model = store_load_student_model(student_id=str(thread_id))
    return {"student_model": model}


def update_student_model(state: TutorState, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Update and persist the student model after the tutor speaks."""

    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id") or "default"

    model = (state.get("student_model") or {}).copy()
    current_topic = state.get("current_topic", "")

    # Merge watcher output (if present) deterministically.
    concept_updates = state.get("concept_updates") or []
    if isinstance(concept_updates, list) and concept_updates:
        model = _merge_concept_updates(model=model, updates=concept_updates, current_topic=current_topic)
    turn_log = model.get("turn_log")
    if not isinstance(turn_log, list):
        turn_log = []

    last_user = ""
    last_ai = ""
    for msg in reversed(state.get("messages", []) or []):
        if not last_ai and isinstance(msg, AIMessage):
            last_ai = msg.content
        if not last_user and isinstance(msg, HumanMessage):
            last_user = msg.content
        if last_user and last_ai:
            break

    turn_log.append({"user": last_user, "assistant": last_ai})
    model["turn_log"] = turn_log[-50:]

    # Persist last tutor hint for debugging/inspection.
    hint = state.get("tutor_hint")
    if isinstance(hint, str) and hint.strip():
        model["last_tutor_hint"] = hint.strip()

    store_save_student_model(student_id=str(thread_id), model=model)
    return {"student_model": model}

# ============================================================================
# Chat Node (Friendly teacher conversation)
# ============================================================================

CHAT_PROMPT = """You are an adaptive tutor who tracks student understanding and guides learning effectively.

Current topic: {current_topic_context}

STUDENT PROFICIENCY DATA:
{student_snapshot}

RECENT OBSERVATION:
{tutor_hint}

YOUR ROLE:
1. Answer the student's question clearly and directly
2. Use the proficiency data to guide what happens next:
   - If concepts show low proficiency (<40%): Offer review or simpler explanation
   - If concepts show medium proficiency (40-70%): Suggest hands-on practice or examples
   - If concepts show high proficiency (>70%): Acknowledge mastery and suggest next topic
3. After answering, ALWAYS include a brief next-step suggestion based on proficiency

RESPONSE PATTERN:
[Answer their question] → [Reflect on understanding] → [Suggest next step]

Examples:
- "Great question! [explanation]... You're showing solid understanding here (proficiency ~60%). Want to try a practice problem to reinforce this?"
- "Let me clarify that. [explanation]... I notice this concept is still developing. Should we go through another example together?"
- "Exactly right! [validation]... You've really got this down. Ready to move on to [next topic]?"

Keep responses natural and conversational (2-3 paragraphs). Be encouraging and supportive.
"""

def chat(state: TutorState) -> Dict[str, Any]:
    """Chat node.

    Node contract:
      - Read what you need from `state`.
      - Return a dict of updates.

    Here we:
      1) Build an LLM input from the session context + message history
      2) Append the latest user input (student_answer) as a HumanMessage
      3) Ask the model for the next assistant message
      4) Append that message to history (via add_messages reducer)
      5) Clear student_answer so the next run doesn't re-process it

    Why clear student_answer?
      - Each `app.invoke(...)` call should consume exactly one user input.
      - Clearing it avoids accidentally treating old input as new input.
    """

    # Conversation so far (this includes prior assistant messages too)
    conversation = state.get("messages", [])

    # Latest user input for this turn (not automatically part of `messages` unless
    # your UI/CLI appends it). We include it here so the LLM responds to what the
    # user just typed.
    student_text = (state.get("student_answer") or "").strip()

    # System prompt sets the "persona" and constraints for the tutor
    current_topic = state.get("current_topic", "")
    topic_context = f"**{current_topic}**" if current_topic else "General (topic will be inferred)"
    
    snapshot = model_snapshot(state.get("student_model") or {}, current_topic=current_topic)
    hint = (state.get("tutor_hint") or "").strip()
    
    system_msg = CHAT_PROMPT.format(
        current_topic_context=topic_context,
        student_snapshot=snapshot if snapshot else "(No proficiency data yet - first interaction)",
        tutor_hint=hint if hint else "(No recent observations)"
    )

    # Build the full message list for this LLM call.
    # Note: we add the *current* HumanMessage only for this call; we also persist
    # it back into state below so future turns have full context.
    llm_messages: list[Any] = [
        {"role": "system", "content": system_msg},
        *conversation,
    ]
    if student_text:
        llm_messages.append(HumanMessage(content=student_text))

    # IMPORTANT:
    # langchain_openai.ChatOpenAI accepts a list of messages.
    # We provide a system message plus the conversation history.
    response = llm.invoke(llm_messages)

    return {
        # Persist BOTH the user's latest message and the assistant response.
        # Because `messages` uses add_messages reducer, this list will be appended
        # onto the existing history.
        "messages": (
            ([HumanMessage(content=student_text)] if student_text else [])
            + [AIMessage(content=response.content)]
        ),

        # Mark the input as consumed for this turn
        "student_answer": "",
    }

# ============================================================================
# Graph Construction
# ============================================================================

def create_tutor_graph():
    """Build and compile the LangGraph app."""

    graph = StateGraph(TutorState)

    # Register node functions
    graph.add_node("router", router)
    graph.add_node("load_student_model", load_student_model)
    graph.add_node("chat", chat)
    graph.add_node("observe_learning", observe_learning)
    graph.add_node("update_student_model", update_student_model)

    # This tells LangGraph where to start each run
    graph.set_entry_point("router")

    # Conditional edge: router returns {"next": "load_student_model"}
    graph.add_conditional_edges(
      "router",
      lambda state: state["next"],
      {
        "load_student_model": "load_student_model",
      },
    )

    # Load external model, then chat, then persist updates.
    graph.add_edge("load_student_model", "chat")
    graph.add_edge("chat", "observe_learning")
    graph.add_edge("observe_learning", "update_student_model")

    # After update, stop. Control returns to your UI/CLI to collect next user input.
    graph.add_edge("update_student_model", END)

    # Checkpointer stores state per "thread" (conversation). MemorySaver is in-memory.
    # In a real app you’d use SQLite/Postgres checkpointers for persistence.
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# ============================================================================
# App Export
# ============================================================================

# Export a compiled graph app so other scripts can import and invoke it.

app = create_tutor_graph()

if __name__ == "__main__":
    print("✅ Tutor agent loaded successfully")
    # `get_graph()` introspects node/edge structure (useful for debugging)
    print("Graph has nodes:", app.get_graph().nodes.keys())