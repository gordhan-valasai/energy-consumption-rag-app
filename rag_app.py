"""
RAG Chat Application for Energy Consumption Data
Streamlit-based interactive Q&A system with vector embeddings

IMPORTANT: This is a Streamlit application. Run it with:
    streamlit run rag_app.py
    
DO NOT run it directly with: python rag_app.py
"""

import sys

# Check if running as Streamlit app
if 'streamlit' not in sys.modules:
    try:
        import streamlit as st
    except ImportError:
        print("=" * 80)
        print("ERROR: This is a Streamlit application!")
        print("=" * 80)
        print("\nTo run this app, use:")
        print("  streamlit run rag_app.py")
        print("\nNOT: python rag_app.py")
        print("\nInstall Streamlit if needed:")
        print("  pip install streamlit")
        print("=" * 80)
        sys.exit(1)

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Optional
import json

# RAG components - with error handling
try:
    # Text splitting (moved to separate package in LangChain v0.1+)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # Embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    # Vector store
    from langchain_community.vectorstores import Chroma
    # RAG chain (updated API)
    from langchain.chains import RetrievalQA
    # Prompts and documents
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError as e:
    # Try fallback imports for older LangChain versions
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.prompts import PromptTemplate
        from langchain.schema import Document
    except ImportError:
        st.error(f"❌ Missing required packages. Please install:")
        st.code("pip install -r requirements_rag.txt", language="bash")
        st.error(f"Error: {e}")
        st.error("\n**Note**: Make sure you have installed:")
        st.code("pip install langchain-text-splitters", language="bash")
        st.stop()

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


class EnergyDataRAG:
    """RAG engine for energy consumption data."""
    
    def __init__(self, use_openai: bool = True, openai_api_key: Optional[str] = None):
        """
        Initialize RAG engine.
        
        Args:
            use_openai: If True, use OpenAI embeddings/LLM. If False, use HuggingFace (free)
            openai_api_key: OpenAI API key (required if use_openai=True)
        """
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        
        if use_openai:
            if not openai_api_key:
                raise ValueError("OpenAI API key required when use_openai=True")
            os.environ['OPENAI_API_KEY'] = openai_api_key
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        else:
            # Use free HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            # For LLM, still need OpenAI or use local model
            # For demo, we'll use OpenAI but you can replace with Ollama/Local
            if openai_api_key:
                os.environ['OPENAI_API_KEY'] = openai_api_key
                self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
            else:
                st.warning("⚠️ Using HuggingFace embeddings but OpenAI API key needed for LLM. Please provide API key.")
                self.llm = None
        
        self.vectorstore = None
        self.qa_chain = None
    
    def load_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load energy data from CSV."""
        return pd.read_csv(csv_path)
    
    def dataframe_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame to LangChain Documents.
        Creates rich text representations of the data.
        """
        documents = []
        
        # Create summary document
        summary_text = f"""
ENERGY CONSUMPTION DATASET SUMMARY:
- Total rows: {len(df)}
- Countries: {df['country'].nunique() if 'country' in df.columns else 'N/A'}
- Years: {df['year'].min() if 'year' in df.columns else 'N/A'} to {df['year'].max() if 'year' in df.columns else 'N/A'}
- Columns: {', '.join(df.columns.tolist())}
"""
        documents.append(Document(page_content=summary_text, metadata={"type": "summary"}))
        
        # Create per-country summaries
        if 'country' in df.columns:
            for country in df['country'].unique():
                country_data = df[df['country'] == country].sort_values('year' if 'year' in df.columns else df.columns[0])
                
                country_text = f"""
COUNTRY: {country}
"""
                if 'year' in country_data.columns and 'total_energy_twh' in country_data.columns:
                    latest_year = country_data['year'].max()
                    latest_value = country_data[country_data['year'] == latest_year]['total_energy_twh'].iloc[0] if len(country_data[country_data['year'] == latest_year]) > 0 else None
                    
                    country_text += f"""
- Latest year data: {latest_year}
- Latest energy consumption: {latest_value:.2f} TWh (if available)
- Years of data: {country_data['year'].min()} to {country_data['year'].max()}
- Number of data points: {len(country_data)}
"""
                    
                    # Add per-capita if available
                    if 'energy_per_capita_twh' in country_data.columns:
                        latest_pc = country_data[country_data['year'] == latest_year]['energy_per_capita_twh'].iloc[0] if len(country_data[country_data['year'] == latest_year]) > 0 else None
                        if pd.notna(latest_pc):
                            country_text += f"- Energy per capita: {latest_pc:.4f} TWh\n"
                    
                    # Add growth rate if we have 2000 and latest year
                    if 2000 in country_data['year'].values and latest_year >= 2000:
                        val_2000 = country_data[country_data['year'] == 2000]['total_energy_twh'].iloc[0] if len(country_data[country_data['year'] == 2000]) > 0 else None
                        if pd.notna(val_2000) and pd.notna(latest_value) and val_2000 > 0:
                            growth_rate = ((latest_value / val_2000) ** (1 / (latest_year - 2000)) - 1) * 100
                            country_text += f"- Annual growth rate (2000-{latest_year}): {growth_rate:.2f}%\n"
                
                # Add data quality info if available
                if 'data_quality_flag' in country_data.columns:
                    flags = country_data['data_quality_flag'].value_counts().to_dict()
                    flag_labels = {0: 'original', 1: 'interpolated', 2: 'missing_block', 3: 'removed_outlier'}
                    country_text += "\nData Quality:\n"
                    for flag, count in flags.items():
                        label = flag_labels.get(flag, f'flag_{flag}')
                        country_text += f"- {label}: {count} data points\n"
                
                documents.append(Document(
                    page_content=country_text,
                    metadata={"type": "country", "country": country}
                ))
        
        # Create time-series documents (by year)
        if 'year' in df.columns:
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year]
                
                year_text = f"""
YEAR: {year}
"""
                if 'country' in year_data.columns and 'total_energy_twh' in year_data.columns:
                    top_countries = year_data.nlargest(10, 'total_energy_twh')
                    year_text += "\nTop 10 Energy Consumers:\n"
                    for idx, row in top_countries.iterrows():
                        country = row['country']
                        energy = row['total_energy_twh']
                        year_text += f"- {country}: {energy:.2f} TWh\n"
                    
                    # Global total
                    global_total = year_data['total_energy_twh'].sum()
                    year_text += f"\nGlobal Total: {global_total:.2f} TWh\n"
                
                documents.append(Document(
                    page_content=year_text,
                    metadata={"type": "year", "year": year}
                ))
        
        return documents
    
    def create_vectorstore(self, documents: List[Document], persist_directory: str = "./chroma_db"):
        """Create vector store from documents."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        return self.vectorstore
    
    def create_qa_chain(self, k: int = 4):
        """Create QA chain for question answering."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        
        if not self.llm:
            raise ValueError("LLM not initialized. Please provide OpenAI API key.")
        
        # Custom prompt template
        template = """You are an expert data analyst specializing in global energy consumption data.
Use the following pieces of context from the energy consumption dataset to answer the question.
If you don't know the answer, say that you don't know. Don't make up an answer.

Context: {context}

Question: {question}

Provide a detailed answer with specific numbers and citations. Format your response clearly.
Answer:"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> Dict:
        """Query the RAG system."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call create_qa_chain first.")
        
        # Try both "query" and "question" keys (LangChain API changed)
        try:
            result = self.qa_chain({"query": question})
        except (KeyError, TypeError):
            try:
                result = self.qa_chain({"question": question})
            except Exception as e:
                # Fallback: invoke directly
                result = self.qa_chain.invoke({"query": question})
        
        return result


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Energy Data Chat",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚡ Energy Consumption Data Chat")
    st.markdown("Ask questions about global energy consumption data using natural language!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        use_openai = st.radio(
            "Embedding Model",
            ["OpenAI (Recommended)", "HuggingFace (Free)"],
            help="OpenAI provides better quality but requires API key. HuggingFace is free but slower."
        )
        use_openai_bool = use_openai == "OpenAI (Recommended)"
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for LLM. Get one at https://platform.openai.com/api-keys",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        
        # Data file selection
        st.header("📊 Data Source")
        data_file = st.selectbox(
            "Select data file",
            [
                "output/cleaned_energy_data.csv",
                "output_robust/cleaned_energy_data.csv",
                "raw_energy_data.csv"
            ],
            help="Choose which processed dataset to use"
        )
        
        # Load data button
        if st.button("🔄 Load Data & Initialize RAG", type="primary"):
            with st.spinner("Loading data and creating embeddings..."):
                try:
                    # Check if file exists
                    if not os.path.exists(data_file):
                        st.error(f"❌ File not found: {data_file}")
                        st.info("💡 Run the processing pipeline first to generate cleaned data.")
                    else:
                        # Initialize RAG
                        rag_engine = EnergyDataRAG(
                            use_openai=use_openai_bool,
                            openai_api_key=openai_api_key if openai_api_key else None
                        )
                        
                        # Load data
                        df = rag_engine.load_data_from_csv(data_file)
                        st.session_state.df = df
                        
                        # Convert to documents
                        documents = rag_engine.dataframe_to_documents(df)
                        
                        # Create vector store
                        persist_dir = f"./chroma_db_{data_file.replace('/', '_').replace('.csv', '')}"
                        vectorstore = rag_engine.create_vectorstore(documents, persist_dir)
                        st.session_state.vectorstore = vectorstore
                        
                        # Create QA chain
                        qa_chain = rag_engine.create_qa_chain(k=4)
                        st.session_state.qa_chain = qa_chain
                        st.session_state.rag_engine = rag_engine
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Loaded {len(df)} rows from {data_file}")
                        st.info(f"📚 Created {len(documents)} document chunks")
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        # Display data info
        if st.session_state.data_loaded and 'df' in st.session_state:
            st.header("📈 Data Info")
            df = st.session_state.df
            st.metric("Rows", len(df))
            st.metric("Countries", df['country'].nunique() if 'country' in df.columns else 'N/A')
            st.metric("Years", f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A')
    
    # Main chat interface
    if not st.session_state.data_loaded:
        st.info("👈 Please load data from the sidebar to start chatting!")
        
        # Show example questions
        st.header("💡 Example Questions")
        example_questions = [
            "What was the highest energy consumption year globally?",
            "Which country had the fastest growth rate from 2000 to 2024?",
            "What are the top 5 energy consuming countries in 2024?",
            "Show me energy consumption trends for China",
            "Which countries have missing data?",
            "What is the global total energy consumption in 2020?",
            "Compare energy consumption between USA and China",
            "What is the average energy per capita for European countries?"
        ]
        
        for q in example_questions:
            st.markdown(f"- {q}")
    
    else:
        # Chat interface
        st.header("💬 Chat with Your Data")
        
        # Display chat history
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.expander(f"❓ {question}", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Answer:**\n{answer}")
                if sources:
                    st.markdown("**Sources:**")
                    for j, source in enumerate(sources[:3], 1):  # Show top 3 sources
                        st.text(f"{j}. {source.page_content[:200]}...")
        
        # Question input
        question = st.text_input(
            "Ask a question about the energy data:",
            placeholder="e.g., What was the highest energy consumption year globally?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            ask_button = st.button("🚀 Ask", type="primary")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and question:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_engine.query(question)
                    answer = result['result']
                    sources = result.get('source_documents', [])
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer, sources))
                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        # Data preview
        with st.expander("📊 Preview Data"):
            if 'df' in st.session_state:
                st.dataframe(st.session_state.df.head(20))


if __name__ == "__main__":
    # Check if running via streamlit
    if len(sys.argv) > 0 and 'streamlit' in sys.argv[0]:
        main()
    else:
        print("=" * 80)
        print("⚠️  This is a Streamlit application!")
        print("=" * 80)
        print("\nTo run this app, use:")
        print("  streamlit run rag_app.py")
        print("\nNOT: python rag_app.py")
        print("\nThe app will open in your browser automatically.")
        print("=" * 80)
        sys.exit(1)

