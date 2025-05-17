# Bignalytics RAG Chatbot 🚀

An intelligent Retrieval-Augmented Generation (RAG) based chatbot for Bignalytics Educational Institute, focused on answering user questions about courses, fees, batches, and placements.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone ""
cd bignalytics-rag-chatbot

# Create new environment (example: biganalytics-env)
conda create -n biganalytics python=3.10 -y

# Activate the environment
conda activate biganalytics-env

# Install Python Dependencies
pip install -r requirements.txt

# LLM Setup Options

## Option 1: Use Hugging Face (Recommended)

# Create a .env file and set your Hugging Face API key:
# HF_API_KEY=your_huggingface_api_key_here
# HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# HF_MODEL_NAME=google/gemma-2b-it
# USE_HUGGINGFACE=true

## Option 2: Use Ollama (Fallback)

# Install Ollama following the instructions for your OS:
# 👉 https://ollama.com/download

# Pull Gemma 3 1b Model
# Once Ollama is installed and running, open terminal and run:
ollama run gemma3:1b
# ✅ This will download the Gemma 3 1b model required for query transformation and answer generation.

# Optional: Install additional models
ollama pull phi:2.7b-chat-v2-q4_0
```

## 🔄 Hugging Face Integration

This project now supports Hugging Face models for both embeddings and text generation:

### Embedding Models

The system uses Hugging Face sentence-transformers for creating embeddings:

- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch processing for improved performance
- Normalized embeddings for better similarity search
- Configurable via `HF_EMBEDDING_MODEL` environment variable

### LLM Options

The RAG pipeline supports multiple LLM providers in this priority order:

1. **Groq API** (if `USE_GROQ=true` and API key provided)
2. **Hugging Face**:
   - HF Inference API (cloud-based, requires `HF_API_KEY`)
   - Local HF models (offline use, set `USE_LOCAL_HF=true`)
3. **Ollama** (fallback option if other methods fail)

### Environment Variables

Create a `.env` file with these options:

```
# Hugging Face Configuration
HF_API_KEY=your_huggingface_api_key_here
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_MODEL_NAME=google/gemma-2b-it

# LLM Provider Selection (true/false)
USE_GROQ=false
USE_HUGGINGFACE=true
USE_LOCAL_HF=false  # Set to true to use local HF models instead of API
USE_GPU=false  # Set to true to use CUDA GPU for local models

# Groq Configuration (optional)
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

- Build your ChromaDB vector store with `python src/build_chroma_vectorstore.py`
- Run the Streamlit application with `python src/app.py`
- Your console will display helpful messages about the initialization steps.

