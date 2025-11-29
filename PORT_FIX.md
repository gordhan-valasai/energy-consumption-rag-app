# Port Conflict Fix

## Issue
Port 8501 was already in use, preventing the Streamlit app from starting.

## Solutions Implemented

### 1. Fixed Config Conflict
- **Problem**: `enableCORS = false` and `enableXsrfProtection = true` are incompatible
- **Fix**: Removed `enableCORS = false` from `.streamlit/config.toml`
- **Result**: Streamlit will use default CORS settings compatible with XSRF protection

### 2. Enhanced Launcher Script
- **Problem**: Script didn't handle port conflicts
- **Fix**: Updated `run_rag_app.sh` to:
  - Detect if port 8501 is in use
  - Offer to kill the existing process OR use port 8502
  - Automatically use port 8502 if user chooses option 2

## Quick Fixes

### Option 1: Kill Existing Process
```bash
# Find and kill process on port 8501
lsof -ti:8501 | xargs kill -9

# Then run the app
streamlit run rag_app.py
```

### Option 2: Use Different Port
```bash
# Run on port 8502
streamlit run rag_app.py --server.port 8502
```

### Option 3: Use Updated Launcher (Recommended)
```bash
./run_rag_app.sh
# The script will automatically detect and handle port conflicts
```

## What Changed

1. **`.streamlit/config.toml`**:
   - Removed `enableCORS = false` (incompatible with XSRF protection)
   - Kept `enableXsrfProtection = true` for security

2. **`run_rag_app.sh`**:
   - Added port conflict detection
   - Added interactive choice for handling conflicts
   - Added automatic fallback to port 8502

## Next Steps

Run the app again:
```bash
./run_rag_app.sh
```

The script will now handle port conflicts automatically! 🎉

