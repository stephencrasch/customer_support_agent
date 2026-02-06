# Adaptive Tutor Workspace Guide

This repo contains a few experiments. The maintained “happy path” today is the
diagnostic tutor in `agents_diagnostic.py`, driven by `chat_interactive.py`.

## 📁 Project Structure (Current)

```
customer_support_agent/
├── agents.py                    # Stable shim: re-exports the current graph (`app`)
├── agents_diagnostic.py         # Diagnostic-driven tutor graph (LangGraph)
├── chat_interactive.py          # Interactive CLI for `agents.py`
│
├── prompts.py                   # Centralized prompt text/templates
├── knowledge_graph.py           # Minimal DAG + proficiency tracking
├── graph_persistence.py         # Saves per-thread knowledge graphs (JSON files)
├── study_tools.py               # Small helper “tools” for the tutor
│
├── tutor_agent.py               # Smaller, newer LangGraph tutor (JSON student model)
├── chat_tutor.py                # Interactive CLI for `tutor_agent.py`
├── student_store.py             # JSON-backed persistence for `tutor_agent.py`
│
└── requirements.txt             # Runtime deps
```

## 🎯 Diagnostic Tutor Overview

Core loop:
`ask → diagnose → teach (if needed) → ask ...`

Key pieces:
- Persistent knowledge graph per learner (stored as JSON in `.knowledge_graphs/`)
- Proficiency-based followups (don’t re-teach what’s mastered)
- “Dialogue update” pass to capture evidence that shows up in normal conversation

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Set OPENAI_API_KEY (see .env.example)
python chat_interactive.py

# Alternate demo:
python chat_tutor.py
```

## 🧾 Local Data (Ignored by Git)

These are generated at runtime and should not be committed:
- `.knowledge_graphs/` (per-thread graph snapshots)
- `student_store.json` (tutor_agent demo state)
- `learning_sessions.db` (older persistence experiment)

## 🎓 Next Steps

- Decide which tutor path is “primary” long-term (`agents_diagnostic.py` vs `tutor_agent.py`)
- Trim or update legacy tests
- Add a simple graph viewer (CLI or small web UI)
