"""
Diagnostic-Driven Adaptive Learning System
Learns about students by asking questions, not by pre-building curriculum.
"""
from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Annotated, Literal, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from knowledge_graph import KnowledgeGraph, KnowledgeNode, NodeStatus
from prompts import (
    TEACHER_PROMPT,
    PROGRESSOR_PROMPT,
    GRADER_PROMPT,
    DIALOGUE_UPDATE_PROMPT_TEMPLATE,
    ROUTER_PROMPT_TEMPLATE,
    CHAT_WITH_STUDENT_PROMPT_TEMPLATE,
    ASK_DIAGNOSTIC_QUESTION_PROMPT_TEMPLATE,
    DIAGNOSE_ANSWER_PROMPT_TEMPLATE,
    TEACH_CONCEPT_PROMPT_TEMPLATE,
    LESSON_RECORD_PROMPT_TEMPLATE,
    CHAT_SYS_PROMPT,
    CHAT_CONTEXT_PROMPT_TEMPLATE,
)
from graph_persistence import save_knowledge_graph
from langchain_core.runnables import RunnableConfig
from study_tools import query_learning_progress, canonicalize_concepts

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)


def dialogue_update(state: "StudyState", config: RunnableConfig = None) -> Dict[str, Any]:
    """Post-turn updater: optionally update the knowledge graph from conversation.

    Intended wiring: any tutor-response node -> dialogue_update -> END.

    This node:
    - applies at most once per unique student message
    - does not emit messages
    """

    student_text = (state.get("student_answer") or "").strip()
    if not student_text:
        return {"next": "end"}

    last_applied = (state.get("last_dialogue_update_user_text") or "").strip()
    if last_applied and last_applied == student_text:
        return {"next": "end"}

    tutor_text = (state.get("last_tutor_message") or "").strip()

    # Load graph (or create if missing).
    try:
        graph = KnowledgeGraph.from_json(state["knowledge_graph"]) if state.get("knowledge_graph") else KnowledgeGraph(
            topic=f"{state['user_id']}_knowledge"
        )
    except (TypeError, ValueError, JSONDecodeError):
        graph = KnowledgeGraph(topic=f"{state['user_id']}_knowledge")

    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    existing_ids = list(graph.nodes.keys())

    prompt = DIALOGUE_UPDATE_PROMPT_TEMPLATE.format(
        ultimate_goal=state["ultimate_goal"],
        current_focus=display_topic(focus_id),
        existing_concepts_json=json.dumps(existing_ids, ensure_ascii=False),
        student_text=student_text,
        tutor_text=tutor_text,
    )

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        data = extract_json(resp.content)
    except (JSONDecodeError, TypeError, ValueError):
        data = {}

    should_update = bool(data.get("should_update"))
    raw_updates = data.get("proficiency_updates") if isinstance(data.get("proficiency_updates"), dict) else {}
    raw_new = data.get("new_concepts") if isinstance(data.get("new_concepts"), list) else []

    raw_concepts: list[str] = []
    raw_concepts.extend([str(k) for k in raw_updates.keys() if isinstance(k, str) and k.strip()])
    raw_concepts.extend([str(x) for x in raw_new if isinstance(x, str) and x.strip()])
    raw_concepts = list(dict.fromkeys([c.strip() for c in raw_concepts if c.strip()]))

    if not should_update or not raw_concepts:
        return {
            "last_dialogue_update_user_text": student_text,
            "student_answer": "",
            "next": "end",
        }

    canon_map = canonicalize_concepts_map(
        ultimate_goal=state["ultimate_goal"],
        existing_concept_ids=existing_ids,
        raw_concepts=raw_concepts,
    )

    # Apply proficiency updates.
    for raw, prof in raw_updates.items():
        if not isinstance(raw, str) or raw not in canon_map:
            continue
        cid = canon_map[raw]
        try:
            p = float(prof)
        except (TypeError, ValueError):
            continue
        p = max(0.0, min(100.0, p))

        if cid not in graph.nodes:
            graph.add_node(
                KnowledgeNode(
                    id=cid,
                    name=display_topic(cid),
                    description=f"Understanding of {display_topic(cid)}",
                    proficiency=p,
                    status=NodeStatus.IN_PROGRESS,
                    attempts=0,
                )
            )
        else:
            node = graph.get_node(cid)
            if node:
                node.proficiency = p

    # Ensure new concepts exist (even if no proficiency given).
    for raw in raw_new:
        if not isinstance(raw, str) or raw not in canon_map:
            continue
        cid = canon_map[raw]
        if cid and cid not in graph.nodes:
            graph.add_node(
                KnowledgeNode(
                    id=cid,
                    name=display_topic(cid),
                    description=f"Understanding of {display_topic(cid)}",
                    proficiency=10.0,
                    status=NodeStatus.IN_PROGRESS,
                    attempts=0,
                )
            )

    new_json = graph.to_json()

    # Best-effort persistence for inspection.
    try:
        cfg = config or {}
        thread_id = cfg.get("configurable", {}).get("thread_id") or cfg.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            save_knowledge_graph(thread_id=thread_id, knowledge_graph_json=new_json)
    except (OSError, TypeError, ValueError):
        pass

    return {
        "knowledge_graph": new_json,
        "last_dialogue_update_user_text": student_text,
        "student_answer": "",
    "next": "end",
    }


# ============================================================================
# State Definition
# ============================================================================

class StudyState(TypedDict):
    """Minimal state for diagnostic teaching."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    knowledge_graph: str              # Builds as we discover what student knows
    ultimate_goal: str                # What they want to learn
    current_focus: str                # What we're testing/teaching RIGHT NOW
    student_answer: str
    last_score: int                   # Track last score to inform choice
    awaiting_choice: bool             # legacy (kept for backward compatibility)
    last_tutor_message: str           # last assistant message content (for dialogue evidence)
    last_dialogue_update_user_text: str  # last user text we applied dialogue updates for
    next: Literal["ask", "diagnose", "teach", "chat", "followup", "end"]


# ============================================================================
# Helper Functions
# ============================================================================

def extract_json(content: str) -> dict:
    """Extract JSON from LLM response (handles markdown blocks)."""
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())


def _to_snake_case(text: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _safe_concept_id(label: str | None, ultimate_goal: str) -> str:
    """Very small, deterministic fallback for concept IDs.

    We keep this intentionally dumb: the LLM concept manager should do the real
    normalization. This exists only as a safety net.
    """

    if not label:
        return _to_snake_case(ultimate_goal)
    return _to_snake_case(label)


def canonicalize_concepts_map(
    *,
    ultimate_goal: str,
    existing_concept_ids: list[str],
    raw_concepts: list[str],
) -> dict[str, str]:
    """Return a {raw_label -> canonical_id} mapping.

    Simplicity-first:
    - Prefer the canonicalize_concepts tool (deterministic today, but tool-based).
    - Fall back to simple snake_case if tool output can't be parsed.
    """

    cleaned: list[str] = []
    for c in raw_concepts:
        if isinstance(c, str) and c.strip():
            cleaned.append(c.strip())
    cleaned = list(dict.fromkeys(cleaned))

    if not cleaned:
        return {}

    try:
        tool_out = canonicalize_concepts.invoke(
            {
                "ultimate_goal": ultimate_goal,
                "existing_concept_ids_json": json.dumps(existing_concept_ids, ensure_ascii=False),
                "raw_concepts_json": json.dumps(cleaned, ensure_ascii=False),
            }
        )
        data = extract_json(str(tool_out)) if isinstance(tool_out, str) else {}
        canonical = data.get("canonical") if isinstance(data.get("canonical"), dict) else {}

        out: dict[str, str] = {}
        for raw, cid in canonical.items():
            if not isinstance(raw, str) or not isinstance(cid, str):
                continue
            out[raw] = _to_snake_case(cid)
        return out
    except (JSONDecodeError, TypeError, ValueError):
        return {raw: _safe_concept_id(raw, ultimate_goal) for raw in cleaned}


def normalize_focus_id(focus: str | None, ultimate_goal: str) -> str:
    """Back-compat alias.

    Existing code calls this in many places. It now delegates to the deterministic
    fallback. The dedicated concept manager handles true normalization.
    """

    return _safe_concept_id(focus, ultimate_goal)


def display_topic(focus_id: str) -> str:
    """Convert a focus id to a nice display name."""
    return focus_id.replace("_", " ").strip().title()


# ============================================================================
# Router (LLM Decides)
# ============================================================================

def router(state: StudyState) -> Dict[str, Any]:
    """LLM-based conversational router.

    This picks the next node based on the student's message + current teaching context.
    Output is constrained to JSON for reliability.
    """

    student_text = (state.get("student_answer") or "").strip()
    if not student_text:
        # Kickoff behavior: if we don't have any user text yet, start with an ask.
        # (After a user turn, `chat_interactive.py` always supplies student_answer.)
        return {"next": "ask"}

    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    router_prompt = ROUTER_PROMPT_TEMPLATE.format(
        ultimate_goal=state["ultimate_goal"],
        current_focus=display_topic(focus_id),
    )

    decision = {}
    try:
        resp = llm.invoke([HumanMessage(content=router_prompt)])
        decision = extract_json(resp.content)
    except (JSONDecodeError, TypeError, ValueError):
        decision = {}

    route = str(decision.get("next") or "").strip().lower()
    if route not in {"followup", "diagnose", "ask", "teach", "chat", "end"}:
        # Safe fallback: if it looks like a question, treat as followup; otherwise diagnose.
        route = "followup" if "?" in student_text else "diagnose"

    target_focus = decision.get("target_focus")
    updates: Dict[str, Any] = {"next": route}
    if isinstance(target_focus, str) and target_focus.strip():
        # Canonicalize router-provided focus against existing concept ids.
        existing_ids: list[str] = []
        try:
            if state.get("knowledge_graph"):
                g2 = KnowledgeGraph.from_json(state["knowledge_graph"])
                existing_ids = list(g2.nodes.keys())
        except (JSONDecodeError, TypeError, ValueError):
            existing_ids = []

        try:
            canon = canonicalize_concepts_map(
                ultimate_goal=state["ultimate_goal"],
                existing_concept_ids=existing_ids,
                raw_concepts=[target_focus],
            )
            updates["current_focus"] = canon.get(target_focus) or _safe_concept_id(
                target_focus, state["ultimate_goal"]
            )
        except (KeyboardInterrupt, TimeoutError, ValueError, TypeError, JSONDecodeError):
            updates["current_focus"] = _safe_concept_id(target_focus, state["ultimate_goal"])
    return updates


# ============================================================================
# Ask Node (Diagnostic Questions)
# ============================================================================

def ask_diagnostic_question(state: StudyState) -> Dict[str, Any]:
    """Generate question to probe current focus area."""
    
    graph = KnowledgeGraph.from_json(state["knowledge_graph"]) if state.get("knowledge_graph") else None
    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    focus_node = graph.get_node(focus_id) if graph and focus_id else None

    focus_memory = None
    if focus_node:
        focus_memory = {
            "taught_points": getattr(focus_node, "taught_points", []) or [],
            "misconceptions": getattr(focus_node, "misconceptions", []) or [],
            "last_check_question": getattr(focus_node, "last_check_question", "") or "",
        }

    known_concepts: list[dict[str, Any]] = []
    if graph:
        known_concepts = [
            {
                "concept": n.id,
                "proficiency": n.proficiency,
                "attempts": n.attempts,
            }
            for n in graph.nodes.values()
            if n.proficiency > 0
        ]
    
    focus_display = display_topic(focus_id)
    question_prompt = ASK_DIAGNOSTIC_QUESTION_PROMPT_TEMPLATE.format(
        ultimate_goal=state["ultimate_goal"],
        current_focus=focus_display,
        focus_memory_json=json.dumps(focus_memory, indent=2, ensure_ascii=False) if focus_memory else "None",
        known_concepts_json=(
            json.dumps(known_concepts, indent=2, ensure_ascii=False)
            if known_concepts
            else "Nothing yet - first question"
        ),
    )
    
    response = llm.invoke([
        SystemMessage(content=PROGRESSOR_PROMPT),
        HumanMessage(content=question_prompt)
    ])
    
    try:
        data = extract_json(response.content)
        question = data.get("question", response.content)
    except (JSONDecodeError, TypeError, ValueError):
        question = response.content
    
    msg = f"❓ {question}"
    return {
        "messages": [SystemMessage(content=msg)],
        "last_tutor_message": msg,
        "student_answer": "",
        "next": "end",
    }


# ============================================================================
# Diagnose Node (The Intelligence)
# ============================================================================

def diagnose_answer(state: StudyState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Analyze answer, update knowledge graph, decide next focus."""
    
    # Load or create knowledge graph
    if state.get("knowledge_graph"):
        graph = KnowledgeGraph.from_json(state["knowledge_graph"])
    else:
        graph = KnowledgeGraph(topic=f"{state['user_id']}_knowledge")
    
    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    current_knowledge = (
        [
            {"concept": n.id, "name": n.name, "proficiency": n.proficiency}
            for n in graph.nodes.values()
        ]
        if graph.nodes
        else "Empty - new student"
    )
    diagnostic_prompt = DIAGNOSE_ANSWER_PROMPT_TEMPLATE.format(
        ultimate_goal=state["ultimate_goal"],
        current_focus=display_topic(focus_id),
        student_answer=state["student_answer"],
        current_knowledge_json=json.dumps(current_knowledge, indent=2, ensure_ascii=False)
        if isinstance(current_knowledge, list)
        else current_knowledge,
    )
    
    response = llm.invoke([
        SystemMessage(content=GRADER_PROMPT),
        HumanMessage(content=diagnostic_prompt)
    ])
    
    try:
        data = extract_json(response.content)
    except (JSONDecodeError, TypeError, ValueError):
        # Fallback
        data = {
            "score": 50,
            "feedback": response.content,
            "next_action": "ask",
            "next_focus": state.get("current_focus", state["ultimate_goal"])
        }
    
    # Canonicalize concept ids before touching the graph.
    raw_updates = data.get("proficiency_updates", {}) if isinstance(data.get("proficiency_updates", {}), dict) else {}

    raw_concepts: list[str] = []
    raw_concepts.extend([str(k) for k in raw_updates.keys() if isinstance(k, str) and k.strip()])
    for k in ["gap_detected", "root_cause", "next_focus"]:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            raw_concepts.append(v.strip())

    existing_ids = list(graph.nodes.keys())
    canon = canonicalize_concepts_map(
        ultimate_goal=state["ultimate_goal"],
        existing_concept_ids=existing_ids,
        raw_concepts=raw_concepts,
    )

    def _canon(label: str | None) -> str:
        if isinstance(label, str) and label.strip() and label in canon:
            return canon[label]
        return _safe_concept_id(label, state["ultimate_goal"])

    # Rewrite the grader outputs into canonical ids.
    canonical_updates: dict[str, float] = {}
    for concept, proficiency in raw_updates.items():
        if not isinstance(concept, str) or not concept.strip():
            continue
        cid = _canon(concept)
        try:
            canonical_updates[cid] = float(proficiency)
        except (TypeError, ValueError):
            continue

    data["proficiency_updates"] = canonical_updates
    if isinstance(data.get("gap_detected"), str):
        data["gap_detected"] = _canon(data.get("gap_detected"))
    if isinstance(data.get("root_cause"), str):
        data["root_cause"] = _canon(data.get("root_cause"))
    if isinstance(data.get("next_focus"), str):
        data["next_focus"] = _canon(data.get("next_focus"))

    # Update knowledge graph
    for concept, proficiency in canonical_updates.items():
        if concept not in graph.nodes:
            node = KnowledgeNode(
                id=concept,
                name=display_topic(concept),
                description=f"Understanding of {display_topic(concept)}",
                proficiency=proficiency,
                status=NodeStatus.IN_PROGRESS,
                attempts=1,
            )
            graph.add_node(node)
        else:
            graph.update_proficiency(concept, proficiency)
    
    # Build feedback message
    score = data.get("score", 50)
    feedback_msg = f"""📊 Score: {score}/100

{data.get('feedback', '')}

💡 {data.get('reasoning', '')}"""
    
    # Decide next action and focus
    next_action = data.get("next_action", "ask")
    next_focus = _canon(data.get("next_focus") or state.get("current_focus"))
    
    # Smart decision logic (conversational): decide teach vs ask.
    if score < 40:
        # Very low score - likely missing prerequisites. Prefer teaching.
        if data.get("root_cause") and data.get("root_cause") != state.get("current_focus"):
            next_focus = _canon(data["root_cause"])
            feedback_msg += f"\n\n🎯 I notice you might need to understand {display_topic(next_focus)} first."
        next_action = "teach"
    elif score < 70:
        # Moderate score - teach a clarification.
        next_action = "teach"
    else:
        # Good score - advance.
        next_action = "ask"
    
    # Check if they've mastered the ultimate goal
    ultimate_concept = state["ultimate_goal"].lower().replace(" ", "_")
    if ultimate_concept in graph.nodes:
        ultimate_proficiency = graph.get_node(ultimate_concept).proficiency
        if ultimate_proficiency >= 80:
            next_action = "end"
    
    new_state = {
        "knowledge_graph": graph.to_json(),
        "current_focus": next_focus,
        # Clear the student's answer so the next input is handled conversationally by the router.
        "student_answer": "",
        "last_score": score,
        "next": next_action,
    "messages": [SystemMessage(content=feedback_msg)],
    "last_tutor_message": feedback_msg,
    }

    # Best-effort persistence of the knowledge graph for inspection across restarts.
    try:
        cfg = config or {}
        thread_id = cfg.get("configurable", {}).get("thread_id") or cfg.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            save_knowledge_graph(thread_id=thread_id, knowledge_graph_json=new_state["knowledge_graph"])
    except (OSError, TypeError, ValueError):
        pass

    return new_state


# ============================================================================
# Teach Node
# ============================================================================

def teach_concept(state: StudyState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Provide focused teaching on current gap."""

    # Load or create a knowledge graph (empty string can happen at session start).
    if state.get("knowledge_graph"):
        graph = KnowledgeGraph.from_json(state["knowledge_graph"])
    else:
        graph = KnowledgeGraph(topic=f"{state['user_id']}_knowledge")
    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    focus_node = graph.get_node(focus_id) if focus_id else None

    # Ensure the node exists so we can attach cumulative memory.
    if focus_id and not focus_node:
        focus_node = KnowledgeNode(
            id=focus_id,
            name=display_topic(focus_id),
            description=f"Understanding of {display_topic(focus_id)}",
            proficiency=0.0,
            status=NodeStatus.IN_PROGRESS,
            attempts=0,
        )
        graph.add_node(focus_node)

    # Pull the most recent teacher message (if any) to avoid repeating ourselves.
    last_teacher_msg = ""
    for msg in reversed(state.get("messages", [])):
        msg_content = getattr(msg, "content", "") or ""
        if isinstance(msg_content, str) and msg_content.strip().startswith("📖"):
            last_teacher_msg = msg_content.strip()
            break

    proficiency = float(focus_node.proficiency) if focus_node else 0.0
    last_score = int(state.get("last_score", 0))

    taught_points = list(getattr(focus_node, "taught_points", []) or []) if focus_node else []
    misconceptions = list(getattr(focus_node, "misconceptions", []) or []) if focus_node else []
    last_check_q = (getattr(focus_node, "last_check_question", "") or "") if focus_node else ""

    # A micro-objective makes the lesson progress instead of repeating.
    if proficiency < 40:
        micro_goal = (
            f"After this, the student can define {display_topic(focus_id)} in one sentence "
            f"and name one real-world example."
        )
    elif proficiency < 70:
        micro_goal = (
            f"After this, the student can explain one key mechanism of {display_topic(focus_id)} "
            f"and why it matters for {state['ultimate_goal']}."
        )
    else:
        micro_goal = (
            f"After this, the student can describe one nuance/edge case of {display_topic(focus_id)} "
            f"and how to apply it."
        )

    teach_prompt = TEACH_CONCEPT_PROMPT_TEMPLATE.format(
        topic=display_topic(focus_id),
        ultimate_goal=state["ultimate_goal"],
        proficiency=f"{proficiency:.0f}/100",
        last_score=f"{last_score}/100",
        micro_goal=micro_goal,
        taught_points_json=(
            json.dumps(taught_points, ensure_ascii=False)
            if taught_points
            else "[]"
        ),
        misconceptions_json=(
            json.dumps(misconceptions, ensure_ascii=False)
            if misconceptions
            else "[]"
        ),
        last_check_question=last_check_q if last_check_q else "<none>",
        last_teacher_message=last_teacher_msg if last_teacher_msg else "<no prior lesson>",
    )

    response = llm.invoke([
        SystemMessage(content=TEACHER_PROMPT),
        HumanMessage(content=teach_prompt)
    ])

    content = (response.content or "").strip()
    # Fallback: keep it conversational (no forced quiz question).
    if not content:
        content = (
            f"Let's work on {display_topic(focus_id)} step by step. "
            "What part feels most confusing right now?"
        )

    # Extract a tiny structured record for cumulative memory.
    record_prompt = LESSON_RECORD_PROMPT_TEMPLATE.format(teaching_message=content)

    record: dict = {}
    try:
        record_resp = llm.invoke([HumanMessage(content=record_prompt)])
        record = extract_json(record_resp.content)
    except (JSONDecodeError, TypeError, ValueError):
        record = {}

    if focus_node:
        new_points = [p.strip() for p in (record.get("taught_points") or []) if isinstance(p, str) and p.strip()]
        new_mis = [m.strip() for m in (record.get("misconceptions") or []) if isinstance(m, str) and m.strip()]
        check_q = record.get("check_question") if isinstance(record.get("check_question"), str) else ""

        focus_node.taught_points = (taught_points + new_points)[-8:]
        focus_node.misconceptions = (misconceptions + new_mis)[-4:]
        if check_q and check_q.strip().endswith("?"):
            focus_node.last_check_question = check_q.strip()

    msg = f"📖 {content}"
    new_state = {
        "messages": [SystemMessage(content=msg)],
        "last_tutor_message": msg,
        "knowledge_graph": graph.to_json(),
    }

    # Best-effort persistence of the knowledge graph for inspection across restarts.
    try:
        cfg = config or {}
        thread_id = cfg.get("configurable", {}).get("thread_id") or cfg.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            save_knowledge_graph(thread_id=thread_id, knowledge_graph_json=new_state["knowledge_graph"])
    except (OSError, TypeError, ValueError):
        pass

    return new_state


# ============================================================================
# Offer Choice Node
# ============================================================================

def offer_choice(state: StudyState) -> Dict[str, Any]:
    """Legacy: previously offered an A/B choice (teach vs ask).

    The active graph no longer routes here. Kept only so older persisted states
    (or external callers) don't crash on import.
    """

    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    focus_topic = display_topic(focus_id)
    score = state.get("last_score", 50)
    
    if score < 40:
        # Low score - suggest teaching
        choice_msg = f"""🤔 Based on your answer, it seems like **{focus_topic}** might be new territory.

Would you like me to:
A) **Teach you about {focus_topic}** (I'll explain it clearly)
B) **Ask another question** to better understand what you know

Reply with A or B"""
    elif score < 70:
        # Medium score - either works
        choice_msg = f"""📚 You have some understanding of **{focus_topic}**, but there's room to grow.

Would you like me to:
A) **Teach you more about {focus_topic}** (fill in the gaps)
B) **Ask another question** to keep exploring your knowledge

Reply with A or B"""
    else:
        # High score - suggest advancing (but this shouldn't hit offer_choice usually)
        choice_msg = f"""✨ You're doing well with **{focus_topic}**!

Would you like me to:
A) **Teach you more advanced concepts**
B) **Ask another question** to verify your understanding

Reply with A or B"""
    
    return {
        "messages": [SystemMessage(content=choice_msg)],
    # Keep student's next input for interpreting A/B.
    "awaiting_choice": True,
    "next": "offer_choice",
    }


# ============================================================================
# Follow-up Node (Clarifying questions)
# ============================================================================

def answer_followup(state: StudyState) -> Dict[str, Any]:
    """Answer a student's clarifying question about the current focus.

    This is what makes the tutor feel conversational: we answer, then stop.
    """

    student_text = (state.get("student_answer") or "").strip()
    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])

    # Pull last teacher message and node memory.
    last_teacher_msg = ""
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", "") or ""
        if isinstance(content, str) and content.strip().startswith("📖"):
            last_teacher_msg = content.strip()
            break

    prompt = CHAT_WITH_STUDENT_PROMPT_TEMPLATE.format(
        ultimate_goal=state["ultimate_goal"],
        current_focus=display_topic(focus_id),
        last_tutor_message=(last_teacher_msg if last_teacher_msg else "<none>"),
        student_text=student_text,
    )

    resp = llm.invoke([
        SystemMessage(content=TEACHER_PROMPT),
        HumanMessage(content=prompt),
    ])

    msg = f"💬 {resp.content}"
    out = {
        "messages": [SystemMessage(content=msg)],
        "last_tutor_message": msg,
        "student_answer": "",
        "next": "followup",
    }
    return out


# ============================================================================
# Chat Node (Meta Questions)
# ============================================================================

def chat_with_student(state: StudyState) -> Dict[str, Any]:
    """Conversational chat.

    - If the student is doing small talk / acknowledgement, respond naturally.
    - If they ask about progress or the knowledge graph, tool-call into the graph.
    """

    student_text = (state.get("student_answer") or "").strip()
    if not student_text:
        return {"next": "router"}

    # Deterministic fast-paths for explicit progress/graph requests.
    # This keeps behavior reliable even if the model declines to tool-call.
    lowered = student_text.lower().strip()
    if any(p in lowered for p in [
        "how am i doing",
        "progress",
        "score",
        "what did i learn",
        "what have i learned",
    ]):
        snapshot = query_learning_progress.invoke({
            "knowledge_graph_json": state.get("knowledge_graph") or "",
            "query": "summary",
        })
        msg = f"💬 {snapshot}"
        return {
            "messages": [SystemMessage(content=msg)],
            "last_tutor_message": msg,
            "student_answer": "",
            "next": "router",
        }

    if "knowledge graph" in lowered and any(k in lowered for k in ["show", "print", "view", "see", "dump", "json"]):
        graph_dump = query_learning_progress.invoke({
            "knowledge_graph_json": state.get("knowledge_graph") or "",
            "query": "graph_json",
        })
        msg = f"💬 {graph_dump}"
        return {
            "messages": [SystemMessage(content=msg)],
            "last_tutor_message": msg,
            "student_answer": "",
            "next": "router",
        }

    tools = [query_learning_progress]
    llm_chat = ChatOpenAI(model="gpt-4o", temperature=0.6).bind_tools(tools)

    sys = SystemMessage(content=CHAT_SYS_PROMPT)

    focus_id = normalize_focus_id(state.get("current_focus"), state["ultimate_goal"])
    context = HumanMessage(
        content=CHAT_CONTEXT_PROMPT_TEMPLATE.format(
            student_text=student_text,
            ultimate_goal=state["ultimate_goal"],
            current_focus=display_topic(focus_id),
            knowledge_graph_exists="yes" if (state.get("knowledge_graph") or "").strip() else "no",
        )
    )

    messages: list[BaseMessage] = [sys, context]

    tool_by_name = {t.name: t for t in tools}
    last_ai: BaseMessage | None = None

    for _ in range(3):
        ai = llm_chat.invoke(messages)
        last_ai = ai
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            break

        from langchain_core.messages import ToolMessage

        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tool_fn = tool_by_name.get(name)
            if not tool_fn:
                continue

            # Ensure the tool always has access to graph JSON from state.
            args = dict(args)
            args.setdefault("knowledge_graph_json", state.get("knowledge_graph") or "")

            tool_result = tool_fn.invoke(args)
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=call.get("id")))

    response_text = (getattr(last_ai, "content", "") or "").strip()
    if not response_text:
        response_text = "Got it. Want to keep going?"

    msg = f"💬 {response_text}"
    out = {
        "messages": [SystemMessage(content=msg)],
        "last_tutor_message": msg,
        "student_answer": "",
        "next": "router",
    }
    return out
# ============================================================================

# ============================================================================
# End Node
# ============================================================================

def end_session(state: StudyState) -> Dict[str, Any]:
    """Celebrate completion."""

    # End can be reached via the router guard (no user input) or via mastery.
    # In both cases, the graph may be empty/invalid early in a session.
    try:
        graph = (
            KnowledgeGraph.from_json(state["knowledge_graph"])
            if (state.get("knowledge_graph") or "").strip()
            else KnowledgeGraph(topic=f"{state.get('user_id', 'user')}_knowledge")
        )
    except (TypeError, ValueError, JSONDecodeError):
        graph = KnowledgeGraph(topic=f"{state.get('user_id', 'user')}_knowledge")

    concepts = list(getattr(graph, "nodes", {}).values())
    mastered = [n for n in concepts if n.proficiency >= 80]
    avg_proficiency = sum(n.proficiency for n in concepts) / len(concepts) if concepts else 0
    
    msg = f"""🎉 Congratulations! You've learned about {state['ultimate_goal']}!

📈 Learning Stats:
- Concepts Mastered: {len(mastered)}/{len(concepts)}
- Average Proficiency: {avg_proficiency:.1f}%

Great work! 🎓"""
    
    return {
        "messages": [SystemMessage(content=msg)],
        "next": "end",
    }


# ============================================================================
# Graph Construction
# ============================================================================

def create_diagnostic_graph(checkpointer=None):
    """Build the diagnostic teaching workflow."""
    graph = StateGraph(StudyState)
    
    # Add nodes
    graph.add_node("router", router)
    graph.add_node("ask", ask_diagnostic_question)
    graph.add_node("diagnose", diagnose_answer)
    graph.add_node("teach", teach_concept)
    graph.add_node("followup", answer_followup)
    graph.add_node("chat", chat_with_student)
    graph.add_node("dialogue_update", dialogue_update)
    graph.add_node("end", end_session)
    
    # Start at router
    graph.set_entry_point("router")
    
    # Router decides where to go (including handling A/B choices)
    graph.add_conditional_edges(
        "router",
        lambda s: s["next"],
        {
            "ask": "ask",
            "diagnose": "diagnose",
            "teach": "teach",
            "chat": "chat",
            "followup": "followup",
            "end": "end",
        }
    )
    
    # Diagnose decides what happens next
    graph.add_conditional_edges(
        "diagnose",
        lambda s: s["next"],
        {
            "teach": "teach",
            "ask": "ask",
            "end": "end",
            # If diagnose ever returns chat (unlikely), route it.
            "chat": "chat",
        },
    )

    # Post-turn dialogue update: capture learning evidence from the student's message.
    # This runs only after a tutor reply (teach/chat/followup) and then stops.
    graph.add_edge("teach", "dialogue_update")
    graph.add_edge("followup", "dialogue_update")
    graph.add_edge("chat", "dialogue_update")
    graph.add_edge("dialogue_update", END)

    # Ask produces the question and stops; the next user input will start a new run.
    graph.add_edge("ask", END)

    # End still stops execution.
    graph.add_edge("end", END)
    
    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# App Export
# ============================================================================

memory = MemorySaver()
app = create_diagnostic_graph(checkpointer=memory)
