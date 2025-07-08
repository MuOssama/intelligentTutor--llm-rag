# RAG PDF Q&A System

A simple Retrieval-Augmented Generation (RAG) application for querying PDF documents using local LLMs. Process research papers and get intelligent answers with source citations.

## Features

- 📄 **PDF Processing**: Extract and chunk text from documents
- 🔍 **Semantic Search**: Find relevant content using embeddings
- 🤖 **Local LLM**: Uses Ollama with Llama 3.2:3B
- 🎯 **Source Attribution**: Provides citations and similarity scores
- 🖥️ **GUI Interface**: Easy-to-use desktop application
- ⚡ **Vector Database**: FAISS or ChromaDB support

## Quick Start

### 1. Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai) installed

### 2. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd rag-application

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r venv/requirements.txt
```

### 3. Setup Ollama
```bash
# Pull the model
ollama pull llama3.2:3b

# Verify installation
ollama list
```

### 4. Create Configuration
```bash
# Create config.py
cat > config.py << 'EOF'
OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_MODEL = "llama3.2:3b"
VECTOR_DB_TYPE = "faiss"
DATA_DIR = "./data"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cpu"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
FAISS_INDEX_TYPE = "IndexFlatIP"
GUI_TITLE = "RAG PDF Q&A System"
GUI_GEOMETRY = "1000x700"

def validate_config(): return []
def get_config_summary(): 
    return {"Vector DB": VECTOR_DB_TYPE, "LLM Model": LLAMA_MODEL, "Device": DEVICE}
EOF
```

### 5. Run Application
```bash
python main.py
```

### 6. First Use
1. Click **"Check LLM"** to verify Ollama connection
2. **Browse** and select a PDF file
3. Click **"Process PDF"** to index the document
4. Switch to **"Query"** tab and ask questions!

**Try asking:**
- "What is the main contribution of this paper?"
- "What methodology was used?"
- "Summarize the key findings"

## Architecture

```
main.py          # Application orchestrator
├── gui.py       # Tkinter interface
├── rag.py       # Document processing & vector search
├── llm.py       # Ollama integration
└── config.py    # Configuration settings
```

## Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `VECTOR_DB_TYPE` | `"faiss"` | Vector database (`"faiss"` or `"chromadb"`) |
| `DEVICE` | `"cpu"` | Processing device (`"cpu"` or `"cuda"`) |
| `CHUNK_SIZE` | `1000` | Text chunk size for processing |
| `SEARCH_TOP_K` | `5` | Number of relevant chunks to retrieve |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum similarity score for results |

## Optional: ChromaDB Installation
```bash
pip install chromadb
# Then set VECTOR_DB_TYPE = "chromadb" in config.py
```

## Troubleshooting

**Ollama Issues:**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
ollama serve
```

**Memory Issues:**
- Reduce `CHUNK_SIZE` to 500
- Set `SEARCH_TOP_K` to 3
- Use smaller embedding model: `"all-MiniLM-L6-v2"`

**PDF Processing Errors:**
- Ensure PDF is not password-protected
- Check file permissions
- Verify PDF contains extractable text

## Performance Tips

- **GPU**: Set `DEVICE = "cuda"` and install `faiss-gpu`
- **Speed**: Use `FAISS_INDEX_TYPE = "IndexHNSWFlat"` for large datasets
- **Quality**: Use `"all-mpnet-base-v2"` embedding model

## Example Output

```
QUESTION: What is the main contribution of this paper?

ANSWER:
The main contribution is a novel neural architecture that reduces 
computational overhead by 40% while maintaining accuracy [Source 1].

SOURCES:
1. Research Paper Title
   Similarity: 0.87 | Pages: 1-15
   Preview: The paper presents a comprehensive analysis...
```

## Dependencies

**Core:**
- `torch` - Deep learning framework
- `sentence-transformers` - Text embeddings
- `faiss-cpu` - Vector similarity search
- `PyMuPDF` - PDF processing
- `requests` - HTTP client

**Optional:**
- `chromadb` - Alternative vector database
- `faiss-gpu` - GPU acceleration

## License

Open source - see LICENSE file for details.