# Flower Shop Customer Support Agent

A simple customer support chatbot for a flower shop, built with Streamlit and LangChain.

## Quick Start for Beginners

### Option 1: One-Command Setup (Easiest)
```bash
# This script will check dependencies and guide you through setup
./run.sh
```

### Option 2: Manual Setup
#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Set Up Environment
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Get your API key from: https://platform.openai.com/account/api-keys
```

#### 3. Run the Simple Version (Recommended for Beginners)
```bash
streamlit run simple_streamlit.py
```

#### 4. Run the Full Version (Advanced Features)
```bash
streamlit run streamlit_frontend.py
```

## What's Different?

### Simple Version (`simple_streamlit.py`)
- ✅ Clean, centered layout
- ✅ Just essential chat functionality
- ✅ Clear error messages and setup instructions
- ✅ Beginner-friendly tips
- ✅ Basic debugging info
- ✅ Simple "Start New Chat" button

### Full Version (`streamlit_frontend.py`)
- 🔧 Advanced 3-column layout
- 🔧 Conversation history management
- 🔧 Summary generation and storage
- 🔧 Complex conversation loading/saving
- 🔧 More features but harder to understand

## Production Readiness Checklist

- [x] Dependencies properly listed in requirements.txt
- [x] Environment variables documented with .env.example
- [x] Sample data files (FAQ.json, inventory.json) provided
- [x] Error handling for missing dependencies
- [x] Clear setup instructions
- [x] Simplified interface for beginners
- [x] Proper imports with fallback handling

## Usage

The chatbot can help with:
- Flower and arrangement information
- Store hours and policies
- Product recommendations
- General customer service questions

## Architecture

- **Frontend**: Streamlit (simple_streamlit.py for beginners)
- **Backend**: LangChain with LangGraph for conversation flow
- **LLM**: OpenAI GPT-4o
- **Vector Store**: ChromaDB for FAQ and inventory search
- **Embeddings**: HuggingFace Stella model for semantic search

## Files

- `simple_streamlit.py` - Beginner-friendly Streamlit interface
- `streamlit_frontend.py` - Full-featured interface
- `chatbot.py` - LangGraph conversation logic
- `tools.py` - Custom tools for knowledge base and recommendations
- `vector_store.py` - ChromaDB integration
- `FAQ.json` - Sample FAQ data
- `inventory.json` - Sample flower inventory
- `requirements.txt` - Python dependencies
- `.env.example` - Environment configuration template