#!/bin/bash
# Simple startup script for beginners

echo "🌸 Starting Flower Shop Chatbot..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "🔑 Please edit .env file and add your OpenAI API key!"
    echo "   Get your key from: https://platform.openai.com/account/api-keys"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies. Please check your internet connection."
        exit 1
    fi
fi

echo "🚀 Starting the simple chatbot interface..."
echo ""
echo "💡 The app will open in your web browser"
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Start the simple streamlit app
streamlit run simple_streamlit.py