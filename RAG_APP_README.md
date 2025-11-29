# Energy Data RAG Chat Application

An interactive Streamlit application that allows you to chat with your energy consumption dataset using natural language questions powered by RAG (Retrieval-Augmented Generation).

## 🚀 Features

- **Natural Language Q&A**: Ask questions in plain English about your energy data
- **Vector Embeddings**: Uses ChromaDB for efficient semantic search
- **Citation Support**: Shows source documents for each answer
- **Multiple Data Sources**: Works with processed CSV files
- **Free Option**: Can use HuggingFace embeddings (no API key needed for embeddings)
- **Chat History**: Maintains conversation context

## 📋 Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** (for LLM - get one at https://platform.openai.com/api-keys)
   - Optional: Can use HuggingFace for embeddings (free)
   - Still need OpenAI API key for the LLM unless using local models

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements_rag.txt
```

2. **Set up OpenAI API key** (optional - can enter in UI):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

## 🎯 Usage

### Start the Application

```bash
streamlit run rag_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step-by-Step

1. **Configure Settings** (Sidebar):
   - Choose embedding model (OpenAI or HuggingFace)
   - Enter OpenAI API key (required for LLM)
   - Select data file to use

2. **Load Data**:
   - Click "🔄 Load Data & Initialize RAG"
   - Wait for embeddings to be created (first time takes longer)

3. **Ask Questions**:
   - Type your question in the input box
   - Click "🚀 Ask"
   - View answer with source citations

### Example Questions

- "What was the highest energy consumption year globally?"
- "Which country had the fastest growth rate from 2000 to 2024?"
- "What are the top 5 energy consuming countries in 2024?"
- "Show me energy consumption trends for China"
- "Which countries have missing data?"
- "What is the global total energy consumption in 2020?"
- "Compare energy consumption between USA and China"
- "What is the average energy per capita for European countries?"

## 🏗️ Architecture

```
rag_app.py
├── EnergyDataRAG (RAG Engine)
│   ├── Load CSV data
│   ├── Convert DataFrame → Documents
│   ├── Create vector embeddings
│   ├── Store in ChromaDB
│   └── Create QA chain
└── Streamlit UI
    ├── Configuration sidebar
    ├── Chat interface
    └── Data preview
```

## 📊 How It Works

1. **Data Loading**: Reads CSV file and converts to structured documents
2. **Document Creation**: Creates rich text representations:
   - Dataset summary
   - Per-country summaries (with growth rates, data quality)
   - Per-year summaries (top consumers, global totals)
3. **Embedding**: Converts documents to vector embeddings
4. **Storage**: Stores embeddings in ChromaDB for fast retrieval
5. **Query**: When you ask a question:
   - Finds most relevant document chunks
   - Passes context + question to LLM
   - Returns answer with citations

## 🔧 Configuration Options

### Embedding Models

- **OpenAI** (Recommended): Better quality, requires API key
- **HuggingFace**: Free, uses `sentence-transformers/all-MiniLM-L6-v2`

### Data Files

- `output/cleaned_energy_data.csv` - Standard processed data
- `output_robust/cleaned_energy_data.csv` - Robust processed data
- `raw_energy_data.csv` - Raw input data

## 💡 Tips

1. **First Run**: Takes longer as it creates embeddings
2. **Subsequent Runs**: Faster if ChromaDB persists (default behavior)
3. **Better Questions**: Be specific, include years/countries when relevant
4. **API Costs**: OpenAI API usage is minimal (only for LLM, not embeddings if using HuggingFace)

## 🐛 Troubleshooting

### "File not found" Error
- Run the processing pipeline first: `python process_energy_data.py`
- Check file path in sidebar

### "OpenAI API Key Required" Error
- Enter API key in sidebar
- Or set environment variable: `export OPENAI_API_KEY="your-key"`

### Slow Performance
- First run is slower (creating embeddings)
- Use HuggingFace embeddings for free option
- Reduce number of documents if dataset is very large

### Memory Issues
- Reduce chunk size in `create_vectorstore()` method
- Use smaller embedding models
- Process data in batches

## 🔒 Privacy & Security

- **Local Processing**: Data stays on your machine
- **API Keys**: Never commit API keys to git
- **Vector Store**: Stored locally in `./chroma_db_*` directories

## 🚀 Advanced Usage

### Custom Prompts

Edit the prompt template in `create_qa_chain()` method:

```python
template = """Your custom prompt here...
Context: {context}
Question: {question}
Answer:"""
```

### Local LLM Support

To use local models (e.g., Ollama):

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Modify `EnergyDataRAG.__init__()` to use local LLM

### Multiple Data Sources

Combine multiple CSV files:

```python
df1 = pd.read_csv('file1.csv')
df2 = pd.read_csv('file2.csv')
df_combined = pd.concat([df1, df2])
documents = rag_engine.dataframe_to_documents(df_combined)
```

## 📈 Future Enhancements

- [ ] Support for PDF reports
- [ ] SQL database integration
- [ ] Multi-turn conversations with memory
- [ ] Export chat history
- [ ] Visualization generation from queries
- [ ] Comparison queries (side-by-side)
- [ ] Data quality warnings in answers

## 🤝 Contributing

Feel free to extend this application:
- Add support for more data formats
- Improve document chunking strategies
- Add more sophisticated retrieval methods
- Integrate with visualization library

## 📝 License

Same as main project.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io) - UI framework
- [LangChain](https://langchain.com) - RAG framework
- [ChromaDB](https://www.trychroma.com) - Vector database
- [OpenAI](https://openai.com) - LLM API

