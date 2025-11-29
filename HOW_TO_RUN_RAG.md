# How to Run the RAG Application

## ⚠️ Important: This is a Streamlit App

**DO NOT run it like this:**
```bash
python rag_app.py  # ❌ WRONG - This will fail!
```

**Run it like this:**
```bash
streamlit run rag_app.py  # ✅ CORRECT
```

## 🚀 Quick Start (3 Methods)

### Method 1: Using Streamlit Command (Recommended)
```bash
streamlit run rag_app.py
```

### Method 2: Using the Launcher Script
```bash
./run_rag_app.sh
```

### Method 3: Using Python Module
```bash
python -m streamlit run rag_app.py
```

## 📋 Prerequisites

1. **Install dependencies first:**
```bash
pip install -r requirements_rag.txt
```

2. **The app will automatically:**
   - Open in your browser at `http://localhost:8501`
   - Show the Streamlit interface
   - Allow you to configure and use the RAG system

## 🔍 Why the Error Occurred

The error you saw:
```
ValueError: failed to parse CPython sys.version
```

This happens because:
1. **You ran it with `python rag_app.py`** instead of `streamlit run rag_app.py`
2. Streamlit apps need the Streamlit runtime to initialize properly
3. The app uses `streamlit` module which isn't available when running directly

## ✅ What I Fixed

1. **Added error detection**: App now detects if it's being run incorrectly
2. **Clear error messages**: Shows exactly how to run it
3. **Import protection**: Better error handling for missing packages
4. **Launcher script**: Easy `./run_rag_app.sh` command

## 🎯 Expected Behavior

When you run `streamlit run rag_app.py`:

1. **Terminal output:**
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. **Browser opens automatically** with the app interface

3. **No errors** - everything works smoothly

## 🐛 Troubleshooting

### "streamlit: command not found"
```bash
pip install streamlit
```

### "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install -r requirements_rag.txt
```

### Port 8501 already in use
```bash
streamlit run rag_app.py --server.port 8502
```

### Browser doesn't open automatically
- Manually go to: `http://localhost:8501`

## 📝 Summary

**Remember**: This is a **web application** built with Streamlit, not a regular Python script. It needs the Streamlit server to run, which is why you use `streamlit run` instead of `python`.

