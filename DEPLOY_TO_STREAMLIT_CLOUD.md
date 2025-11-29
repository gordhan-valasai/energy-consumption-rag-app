# Deploy RAG App to Streamlit Community Cloud

## Prerequisites
- GitHub account
- Code pushed to a GitHub repository

## Step-by-Step Deployment Guide

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Repository name: `energy-consumption-rag-app` (or your preferred name)
4. Description: "RAG Chat Application for Energy Consumption Data Analysis"
5. Choose **Public** (required for free Streamlit Cloud)
6. **DO NOT** initialize with README, .gitignore, or license (we already have these)
7. Click **"Create repository"**

### Step 2: Initialize Git (if not already done)

```bash
# Check if git is initialized
git status

# If not initialized, run:
git init
```

### Step 3: Add All Files

```bash
# Add all files to git
git add .

# Commit the files
git commit -m "Initial commit: RAG app for energy consumption data"
```

### Step 4: Connect to GitHub Repository

```bash
# Add your GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# Verify remote is added
git remote -v
```

### Step 5: Push to GitHub

```bash
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

### Step 6: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/REPO_NAME`
5. Branch: `main`
6. Main file path: `rag_app.py`
7. Click **"Deploy"**

### Step 7: Configure Secrets (for OpenAI API Key)

1. In your Streamlit Cloud app dashboard, click **"Settings"** (⚙️)
2. Go to **"Secrets"** tab
3. Add your OpenAI API key:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

4. Click **"Save"**
5. The app will automatically redeploy

## Important Files for Deployment

Make sure these files are in your repository:

✅ **Required:**
- `rag_app.py` - Main Streamlit app
- `requirements_rag.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration (optional but recommended)

✅ **Recommended:**
- `README.md` - Project documentation
- `.gitignore` - Excludes sensitive files
- `RAG_APP_README.md` - App-specific documentation

❌ **Should NOT be committed:**
- `.env` files (use Streamlit Secrets instead)
- `chroma_db/` directory (will be recreated on deployment)
- API keys or secrets
- Large data files (if not needed)

## Troubleshooting

### "App failed to deploy"
- Check that `requirements_rag.txt` includes all dependencies
- Verify `rag_app.py` is the correct main file path
- Check the deployment logs in Streamlit Cloud dashboard

### "Module not found" errors
- Ensure all packages are in `requirements_rag.txt`
- Check that package versions are compatible

### "API Key not found"
- Add `OPENAI_API_KEY` to Streamlit Secrets
- Restart the app after adding secrets

### "Port already in use"
- This shouldn't happen on Streamlit Cloud (handled automatically)
- If testing locally, use `./run_rag_app.sh` which handles port conflicts

## Quick Commands Reference

```bash
# Check git status
git status

# Add all files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin main

# Check remote repository
git remote -v

# View commit history
git log --oneline
```

## Next Steps After Deployment

1. **Test the deployed app** - Make sure everything works
2. **Share the link** - Your app will have a public URL
3. **Monitor usage** - Check Streamlit Cloud dashboard for usage stats
4. **Update code** - Push changes to GitHub, Streamlit Cloud auto-updates

## Need Help?

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Documentation](https://docs.github.com)
- Check deployment logs in Streamlit Cloud dashboard

