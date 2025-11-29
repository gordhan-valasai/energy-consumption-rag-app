#!/bin/bash
# Quick launcher script for RAG application

echo "🚀 Starting Energy Data RAG Chat Application..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit
fi

# Check if requirements are installed
if [ ! -f ".rag_installed" ]; then
    echo "📦 Installing RAG dependencies..."
    pip install -r requirements_rag.txt
    touch .rag_installed
fi

# Check if port 8501 is in use
PORT=8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port $PORT is already in use."
    echo ""
    echo "Options:"
    echo "  1. Kill the existing process and use port $PORT"
    echo "  2. Use a different port (8502)"
    echo ""
    read -p "Choose option (1 or 2, default: 2): " choice
    choice=${choice:-2}
    
    if [ "$choice" = "1" ]; then
        PID=$(lsof -ti:$PORT)
        echo "🛑 Killing process $PID on port $PORT..."
        kill -9 $PID 2>/dev/null
        sleep 1
        echo "✅ Port $PORT is now free."
    else
        PORT=8502
        echo "📌 Using port $PORT instead."
    fi
fi

# Run the app
echo "🌐 Opening app in browser on port $PORT..."
if [ "$PORT" != "8501" ]; then
    streamlit run rag_app.py --server.port $PORT
else
    streamlit run rag_app.py
fi

