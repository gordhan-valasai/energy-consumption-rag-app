# Post-Deployment Guide

## ✅ Your App is Live!

**App URL**: https://energy-consumption-rag-app-dr6dg9xh4yjiatjdjhgk3d.streamlit.app

## 🔐 Step 1: Add OpenAI API Key

1. Go to: https://share.streamlit.io
2. Click on your app: `energy-consumption-rag-app`
3. Click **"Settings"** (⚙️ icon) in the top right
4. Go to **"Secrets"** tab
5. Add this:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-api-key-here"
```

6. Click **"Save"**
7. App will automatically redeploy (takes 1-2 minutes)

## 🧪 Step 2: Test Your App

After adding the API key and redeployment:

1. Visit: https://energy-consumption-rag-app-dr6dg9xh4yjiatjdjhgk3d.streamlit.app
2. Try these questions:
   - "What was the highest energy consumption year globally?"
   - "Which countries are the top 5 energy consumers?"
   - "Show me energy trends for China"
   - "What is the average energy consumption per capita?"

## 📊 Step 3: Load Data

In the app:
1. Select a data file (or upload your own CSV)
2. Enter your OpenAI API key (or it will use the one from Secrets)
3. Click **"Load Data & Initialize RAG"**
4. Wait for embeddings to be created (first time takes longer)
5. Start asking questions!

## 🔍 Troubleshooting

### "API key not found"
- Make sure you added `OPENAI_API_KEY` to Streamlit Secrets
- Check the key starts with `sk-`
- Restart the app after adding secrets

### "File not found"
- Make sure your data files are in the repository
- Or upload a CSV file through the app interface

### "Module not found"
- Check deployment logs in Streamlit Cloud
- Verify `requirements_rag.txt` includes all dependencies

### App won't load
- Check deployment logs: Streamlit Cloud → Your App → Logs
- Look for error messages
- Common issues: missing dependencies, syntax errors

## 📝 Updating Your App

To update your app:

```bash
# Make changes locally
git add .
git commit -m "Update: description of changes"
git push origin main
```

Streamlit Cloud will automatically redeploy!

## 🔗 Useful Links

- **App**: https://energy-consumption-rag-app-dr6dg9xh4yjiatjdjhgk3d.streamlit.app
- **GitHub Repo**: https://github.com/gordhan-valasai/energy-consumption-rag-app
- **Streamlit Cloud**: https://share.streamlit.io
- **OpenAI API Keys**: https://platform.openai.com/api-keys

## 🎉 Success!

Once everything is working, you can:
- Share the app URL with others
- Embed it in websites
- Use it for data analysis
- Customize it further

## 💡 Tips

1. **First Load**: Takes longer as it creates embeddings
2. **Subsequent Queries**: Much faster (embeddings cached)
3. **Data Files**: Can upload CSV directly in the app
4. **API Costs**: Minimal (only for LLM, not embeddings if using HuggingFace)

## 🆘 Need Help?

- Check deployment logs in Streamlit Cloud
- Review error messages in the app
- Check `RAG_APP_README.md` for detailed documentation

