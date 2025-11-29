#!/bin/bash
# Script to fix GitHub remote URL

echo "🔧 Fixing GitHub Remote Configuration"
echo ""

# Remove existing remote
git remote remove origin 2>/dev/null
echo "✅ Removed old remote"

echo ""
echo "📝 Please provide your GitHub information:"
echo ""

# Get GitHub username
read -p "Enter your GitHub username: " GITHUB_USERNAME

# Get repository name
read -p "Enter repository name (default: energy-consumption-rag-app): " REPO_NAME
REPO_NAME=${REPO_NAME:-energy-consumption-rag-app}

# Add correct remote
echo ""
echo "🔗 Adding remote..."
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# Verify
echo ""
echo "✅ Remote configured:"
git remote -v

echo ""
echo "📋 Next steps:"
echo ""
echo "1. Make sure the repository exists on GitHub:"
echo "   https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
echo ""
echo "2. Push your code:"
echo "   git push -u origin main"
echo ""
echo "   When prompted for password, use a Personal Access Token (not your GitHub password)"
echo "   Get token from: https://github.com/settings/tokens"
echo ""
echo "3. Deploy on Streamlit Cloud:"
echo "   https://share.streamlit.io"
echo ""

