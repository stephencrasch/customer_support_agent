# 🎯 Diagnostic-Driven Teaching System - Architecture

## Core Principle
**Discover knowledge through questions, not pre-built curriculum.**

## Simplified Flow

```
User: "I want to learn LSTMs"
  ↓
[ASK] First diagnostic question about the goal
  ↓
Student answers
  ↓
[DIAGNOSE] (enhanced grading)
  - What do they know? Update graph
  - What gaps exist? 
  - Root cause? (look for patterns)
  - Decision: teach gap OR ask deeper OR advance
  ↓
[TEACH] if gap found
  ↓
[ASK] next question (adaptive)
  ↓
... repeat until goal mastered
```

## State (Minimal)

```python
class StudyState(TypedDict):
    messages: list[BaseMessage]
    user_id: str
    knowledge_graph: str              # Builds as we discover
    ultimate_goal: str                # What they want to learn
    current_focus: str                # What we're testing/teaching now
    student_answer: str
    next: Literal["ask", "diagnose", "teach", "chat", "end"]
```

## Nodes (5 Total)

1. **router** - Simple: has answer? → diagnose, else → ask
2. **ask** - Generate diagnostic question on current_focus
3. **diagnose** - Grade + detect gaps + decide next focus
4. **teach** - Brief explanation of gap
5. **chat** - Handle meta questions
6. **end** - Session complete

## Key Intelligence: Diagnosis Node

The `diagnose` node does all the heavy lifting:
- Updates knowledge graph based on answer
- Detects gaps and patterns
- Decides what to focus on next
- Routes to teach (if gap) or ask (if ready)

## Example

```
Goal: LSTMs
  ↓
Ask: "Explain RNNs"
Answer: "Networks that process sequences"
  ↓
Diagnose:
  - Knows: RNN concept (60%)
  - Gap: Missing mechanics
  - Focus: "rnn_hidden_states"
  - Action: TEACH
  ↓
Teach: Hidden state explanation
  ↓
Ask: "How do RNNs maintain memory?"
Answer: "Through hidden states that..."
  ↓
Diagnose:
  - Knows: RNN mechanics (85%)
  - No gaps detected
  - Focus: "lstm_motivation"
  - Action: ASK (advance)
  ↓
Ask: "Why do we need LSTMs?"
```

## Why This Is Simple

- ✅ No pre-analysis of prerequisites
- ✅ No complex path planning
- ✅ No assessment vs learning split
- ✅ Just: ask → diagnose → teach/ask → repeat
- ✅ Graph builds organically
