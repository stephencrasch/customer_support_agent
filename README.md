# customer_support_agent

Adaptive tutor project built with LangGraph.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provide an OpenAI API key (one of these approaches):
- Create a `.env` with `OPENAI_API_KEY=...` (see `.env.example`)
- Or export it in your shell: `export OPENAI_API_KEY=...`

## Run (Recommended)

Diagnostic tutor:

```bash
python chat_interactive.py
```

Alternate minimal tutor demo:

```bash
python chat_tutor.py
```

## Local Data (Ignored by Git)

Runtime artifacts are intentionally ignored:
- `.knowledge_graphs/` (graph snapshots)
- `student_store.json` (tutor_agent demo persistence)
- `learning_sessions.db` (older experiment)
