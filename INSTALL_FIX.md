# Fix for Missing LangChain Packages

## Problem
```
Error: No module named 'langchain.text_splitter'
```

## Solution

LangChain has reorganized its packages. The text splitter is now in a separate package.

### Install Missing Package

```bash
pip install langchain-text-splitters langchain-core
```

Or install all requirements:
```bash
pip install -r requirements_rag.txt
```

## What Changed

### Old Imports (LangChain < 0.1.0)
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document
```

### New Imports (LangChain >= 0.1.0)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
```

## Updated Requirements

The `requirements_rag.txt` has been updated to include:
- `langchain-text-splitters>=0.0.1`
- `langchain-core` (usually included with langchain, but explicit is better)

## Quick Fix

Run this command:
```bash
pip install langchain-text-splitters langchain-core
```

Then restart the Streamlit app:
```bash
streamlit run rag_app.py
```

## Verification

After installing, you can verify the imports work:
```python
python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter; print('✅ Import successful')"
```

## Fallback Support

The code now includes fallback imports for older LangChain versions, so it should work with both old and new versions.

