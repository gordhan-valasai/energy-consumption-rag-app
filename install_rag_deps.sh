#!/bin/bash
# Quick script to install missing RAG dependencies

echo "📦 Installing RAG dependencies..."
echo ""

# Install the missing packages
pip install langchain-text-splitters langchain-core

echo ""
echo "✅ Installation complete!"
echo ""
echo "You can now run the RAG app with:"
echo "  streamlit run rag_app.py"
echo ""
echo "Or use the launcher:"
echo "  ./run_rag_app.sh"

