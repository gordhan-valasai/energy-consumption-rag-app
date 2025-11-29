#!/bin/bash
# Script to prepare the RAG app for GitHub deployment

echo "🚀 Setting up GitHub deployment for RAG App"
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git repository already initialized"
fi

# Check current status
echo ""
echo "📋 Current git status:"
git status --short | head -10

echo ""
echo "📝 Next steps:"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name it: energy-consumption-rag-app"
echo "   - Make it PUBLIC (required for free Streamlit Cloud)"
echo "   - DO NOT initialize with README"
echo ""
echo "2. Add your GitHub repository as remote:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
echo ""
echo "3. Add and commit all files:"
echo "   git add ."
echo "   git commit -m 'Initial commit: RAG app for energy consumption data'"
echo ""
echo "4. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "5. Deploy on Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Sign in with GitHub"
echo "   - Click 'New app'"
echo "   - Select your repository"
echo "   - Main file: rag_app.py"
echo "   - Click 'Deploy'"
echo ""
echo "📖 Full guide: See DEPLOY_TO_STREAMLIT_CLOUD.md"
echo ""

