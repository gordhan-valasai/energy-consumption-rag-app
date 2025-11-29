# Quick Start Guide - RAG Chat Application

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements_rag.txt
```

### Step 2: Set Up API Key (Optional but Recommended)

**Option A: Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Enter in UI**
- Just leave it blank and enter it in the Streamlit sidebar when you run the app

### Step 3: Run the Application

```bash
streamlit run rag_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📝 First Time Setup

1. **Process your data first** (if not already done):
   ```bash
   python process_energy_data.py
   ```

2. **Start the RAG app**:
   ```bash
   streamlit run rag_app.py
   ```

3. **In the sidebar**:
   - Select data file: `output/cleaned_energy_data.csv`
   - Choose embedding model (HuggingFace is free, OpenAI is better)
   - Enter OpenAI API key (if using OpenAI)
   - Click "🔄 Load Data & Initialize RAG"

4. **Wait for initialization** (first time takes 1-2 minutes)

5. **Start asking questions!**

## 💬 Example First Questions

Try these to get started:

1. "What was the highest energy consumption year globally?"
2. "Which country had the fastest growth rate?"
3. "Show me the top 5 energy consuming countries"
4. "What is China's energy consumption trend?"

## 🎯 What Happens Behind the Scenes

1. **Data Loading**: Reads your CSV file
2. **Document Creation**: Converts data to rich text documents:
   - Country summaries
   - Year summaries  
   - Dataset overview
3. **Embedding**: Creates vector embeddings (semantic representations)
4. **Storage**: Saves to ChromaDB for fast retrieval
5. **Query**: When you ask a question:
   - Finds relevant document chunks
   - Sends to LLM with context
   - Returns answer with citations

## ⚙️ Configuration Options

### Embedding Models

- **OpenAI** (Recommended): Better quality, requires API key
  - Cost: ~$0.0001 per 1K tokens
  - Speed: Fast
  - Quality: Excellent

- **HuggingFace** (Free): No API key needed for embeddings
  - Cost: Free
  - Speed: Slower (runs locally)
  - Quality: Good

**Note**: You still need OpenAI API key for the LLM (the part that generates answers), unless you set up a local model.

### Data Files

Choose from:
- `output/cleaned_energy_data.csv` - Standard processed data
- `output_robust/cleaned_energy_data.csv` - Robust processed data (with quality flags)
- `raw_energy_data.csv` - Raw input data

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install -r requirements_rag.txt
```

### "File not found: output/cleaned_energy_data.csv"
Run the processing pipeline first:
```bash
python process_energy_data.py
```

### "OpenAI API key not found"
- Enter it in the sidebar, OR
- Set environment variable: `export OPENAI_API_KEY="your-key"`

### App is slow
- First run is slower (creating embeddings)
- Subsequent runs are faster (embeddings cached)
- Use HuggingFace for free option (slower but no API costs)

### "Out of memory" error
- Reduce dataset size
- Use smaller embedding model
- Process in batches

## 📊 Understanding the Answers

Each answer includes:
- **Main Answer**: Direct response to your question
- **Sources**: Document chunks used (click to expand)
- **Citations**: References to specific data points

## 🎨 UI Features

- **Sidebar**: Configuration and data info
- **Chat Interface**: Question input and history
- **Data Preview**: Quick view of loaded data
- **Expandable History**: Click to see previous Q&A

## 💡 Pro Tips

1. **Be Specific**: Include years, countries, or metrics in questions
2. **Use Natural Language**: Ask like you're talking to a colleague
3. **Check Sources**: Expand sources to see where answers came from
4. **Clear History**: Use "🗑️ Clear History" to start fresh
5. **Try Different Questions**: The RAG system handles various question types

## 🔄 Updating Data

To use new data:
1. Process new data: `python process_energy_data.py`
2. In RAG app sidebar, select the new file
3. Click "🔄 Load Data & Initialize RAG"
4. New embeddings will be created automatically

## 📈 Next Steps

- Try complex questions with multiple countries/years
- Ask for comparisons
- Request trend analysis
- Explore data quality questions

## 🆘 Need Help?

Check `RAG_APP_README.md` for detailed documentation.

