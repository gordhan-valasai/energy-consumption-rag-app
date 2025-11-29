# GitHub Setup - Step by Step

## ✅ What's Already Done
- ✅ Git repository initialized
- ✅ All files committed (36 files)
- ✅ Branch set to `main`

## 🔧 What You Need to Do

### Step 1: Create GitHub Repository

1. Go to **https://github.com/new**
2. Repository name: `energy-consumption-rag-app` (or your choice)
3. Make it **PUBLIC** (required for free Streamlit Cloud)
4. **DO NOT** check "Initialize with README"
5. Click **"Create repository"**

### Step 2: Get Your GitHub Username

Your GitHub username is the part after `github.com/` in your profile URL.
Example: If your profile is `https://github.com/johnsmith`, your username is `johnsmith`

### Step 3: Set Up Authentication

You have **two options**:

#### Option A: Personal Access Token (Easier)

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click **"Generate new token (classic)"**
3. Name it: `Streamlit Deployment`
4. Select scopes: ✅ **repo** (full control)
5. Click **"Generate token"**
6. **COPY THE TOKEN** (you won't see it again!)

#### Option B: SSH (More Secure)

1. Generate SSH key: `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
3. Use SSH URL instead of HTTPS

### Step 4: Update Remote URL

**Replace `YOUR_USERNAME` with your actual GitHub username:**

```bash
# Remove the incorrect remote
git remote remove origin

# Add correct remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/energy-consumption-rag-app.git

# Verify
git remote -v
```

### Step 5: Push to GitHub

**If using Personal Access Token:**

```bash
git push -u origin main
```

When prompted:
- Username: Your GitHub username
- Password: **Paste your Personal Access Token** (not your GitHub password!)

**If using SSH:**

```bash
# Use SSH URL instead
git remote set-url origin git@github.com:YOUR_USERNAME/energy-consumption-rag-app.git
git push -u origin main
```

### Step 6: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/energy-consumption-rag-app`
5. Branch: `main`
6. Main file path: `rag_app.py`
7. Click **"Deploy"**

### Step 7: Add API Key (After Deployment)

1. In Streamlit Cloud dashboard → **Settings** → **Secrets**
2. Add:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

3. Click **"Save"**
4. App will auto-redeploy

## 🐛 Troubleshooting

### "Authentication failed"
- Make sure you're using a Personal Access Token, not your password
- Check that the token has `repo` scope
- Try using SSH instead

### "Remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### "Repository not found"
- Make sure the repository exists on GitHub
- Check the repository name matches exactly
- Verify it's PUBLIC (not private)

## 📝 Quick Commands

```bash
# Check current remote
git remote -v

# Update remote URL
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git push -u origin main

# Check status
git status
```

## ✅ Verification

After pushing, you should see your files on GitHub at:
`https://github.com/YOUR_USERNAME/energy-consumption-rag-app`

