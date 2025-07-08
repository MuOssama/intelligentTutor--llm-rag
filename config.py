# config.py
"""
Configuration settings for RAG application
"""
import torch
import os
from pathlib import Path

# Vector Database Configuration
VECTOR_DB_TYPE = "chromadb"  # Options: "faiss" or "chromadb"

# LLM Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLAMA_MODEL = "llama3.2:3b"
LLM_TIMEOUT = 240
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9
LLM_NUM_PREDICT = 1024

# RAG Configuration
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DATA_DIR = "rag_data"
SEARCH_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# System Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
MAX_WORKERS = os.cpu_count() or 4

# ChromaDB Configuration
CHROMADB_COLLECTION_NAME = "rag_documents"
CHROMADB_PERSIST_DIRECTORY = "rag_data/chromadb"
CHROMADB_SETTINGS = {
    "anonymized_telemetry": False,
    "allow_reset": True
}

# FAISS Configuration
FAISS_INDEX_TYPE = "IndexFlatIP"  # Options: IndexFlatIP, IndexIVFFlat, IndexHNSWFlat
FAISS_NORMALIZE_EMBEDDINGS = True

# GUI Configuration
GUI_TITLE = "Intellegent Instructor"
GUI_GEOMETRY = "1000x700"
GUI_THEME = "default"

# File Configuration
VECTOR_DB_FILE = "vector_db.index"
CHUNKS_FILE = "chunks.pkl"
METADATA_FILE = "metadata.json"
SUPPORTED_FORMATS = [".pdf"]

# PDF Processing Configuration
PDF_DPI = 150
PDF_MAX_PAGES = 1000
TEXT_MIN_LENGTH = 50
SENTENCE_ENDINGS = ['. ', '.\n', '? ', '!\n', '! ']

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "rag_app.log"
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Network Configuration
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAY = 1

# Performance Configuration
BATCH_SIZE = 32
EMBEDDING_CACHE_SIZE = 1000
INDEX_REBUILD_THRESHOLD = 10000

# Error Messages
ERROR_MESSAGES = {
    "NO_PDF": "Please select a PDF file",
    "PDF_NOT_FOUND": "PDF file not found",
    "LLM_NOT_AVAILABLE": "LLM not available. Please check Ollama setup.",
    "NO_DATABASE": "No database loaded. Please process a PDF first.",
    "NO_RESULTS": "No relevant information found",
    "PROCESSING_ERROR": "Error processing document",
    "NETWORK_ERROR": "Network connection error",
    "CHROMADB_NOT_AVAILABLE": "ChromaDB not available. Please install with: pip install chromadb",
    "VECTOR_DB_ERROR": "Vector database error occurred"
}

# System Prompts
SYSTEM_PROMPT = """You are an expert computer architecture researcher. 
Analyze research papers and provide precise, technical answers.
Always cite sources [Source X] for your claims.
Focus on accuracy and technical depth."""

CONTEXT_PROMPT_TEMPLATE = """
[Source {source_id}]
Paper: {paper_title}
Pages: {page_range}
Similarity: {similarity:.3f}
Content: {text}
"""

# Validation Settings
MIN_QUERY_LENGTH = 3
MAX_QUERY_LENGTH = 500
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000

# Cache Settings
ENABLE_CACHE = True
CACHE_TTL = 3600  # 1 hour in seconds
CACHE_MAX_SIZE = 100

# Development Settings
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE_LOGGING = DEBUG_MODE

# Path Configuration
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / DATA_DIR
LOGS_PATH = BASE_DIR / "logs"
CACHE_PATH = BASE_DIR / "cache"
CHROMADB_PATH = BASE_DIR / CHROMADB_PERSIST_DIRECTORY

# Ensure directories exist
DATA_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)
CACHE_PATH.mkdir(exist_ok=True)
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)

# Model Configuration
EMBEDDING_MODELS = {
    "default": "all-MiniLM-L6-v2",
    "large": "all-mpnet-base-v2",
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"
}

# UI Colors and Styling
UI_COLORS = {
    "primary": "#2196F3",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "error": "#F44336",
    "background": "#F5F5F5"
}

# Feature Flags
FEATURES = {
    "auto_save": True,
    "incremental_indexing": True,
    "query_suggestions": False,
    "export_results": True,
    "multi_pdf_support": False,
    "vector_db_switching": True  # Allow switching between FAISS and ChromaDB
}

# Vector Database Specific Settings
VECTOR_DB_CONFIG = {
    "faiss": {
        "index_type": FAISS_INDEX_TYPE,
        "normalize_embeddings": FAISS_NORMALIZE_EMBEDDINGS,
        "save_embeddings": True
    },
    "chromadb": {
        "collection_name": CHROMADB_COLLECTION_NAME,
        "persist_directory": CHROMADB_PERSIST_DIRECTORY,
        "settings": CHROMADB_SETTINGS,
        "distance_metric": "cosine"
    }
}

def get_config_summary():
    """Get configuration summary for debugging"""
    return {
        "vector_db_type": VECTOR_DB_TYPE,
        "device": DEVICE,
        "cuda_available": USE_CUDA,
        "chunk_size": CHUNK_SIZE,
        "embedding_model": EMBEDDINGS_MODEL,
        "llm_model": LLAMA_MODEL,
        "data_dir": str(DATA_PATH),
        "chromadb_dir": str(CHROMADB_PATH),
        "debug_mode": DEBUG_MODE,
        "similarity_threshold": SIMILARITY_THRESHOLD
    }

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if CHUNK_SIZE < MIN_CHUNK_SIZE or CHUNK_SIZE > MAX_CHUNK_SIZE:
        errors.append(f"CHUNK_SIZE must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE}")
    
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
    
    if SEARCH_TOP_K <= 0:
        errors.append("SEARCH_TOP_K must be positive")
    
    if LLM_TIMEOUT <= 0:
        errors.append("LLM_TIMEOUT must be positive")
    
    if VECTOR_DB_TYPE not in ["faiss", "chromadb"]:
        errors.append(f"VECTOR_DB_TYPE must be 'faiss' or 'chromadb', got '{VECTOR_DB_TYPE}'")
    
    if SIMILARITY_THRESHOLD < 0 or SIMILARITY_THRESHOLD > 1:
        errors.append("SIMILARITY_THRESHOLD must be between 0 and 1")
    
    return errors

def get_vector_db_config():
    """Get vector database specific configuration"""
    return VECTOR_DB_CONFIG.get(VECTOR_DB_TYPE, {})

def switch_vector_db(new_db_type: str):
    """Switch vector database type (for runtime configuration)"""
    global VECTOR_DB_TYPE
    if new_db_type in ["faiss", "chromadb"]:
        VECTOR_DB_TYPE = new_db_type
        print(f"Switched to {new_db_type} vector database")
    else:
        print(f"Invalid vector database type: {new_db_type}")

# Check ChromaDB availability
def check_chromadb_availability():
    """Check if ChromaDB is available"""
    try:
        import chromadb
        return True
    except ImportError:
        return False

CHROMADB_AVAILABLE = check_chromadb_availability()

# Auto-adjust if ChromaDB is not available
if VECTOR_DB_TYPE == "chromadb" and not CHROMADB_AVAILABLE:
    print("Warning: ChromaDB not available, falling back to FAISS")
    VECTOR_DB_TYPE = "faiss"

# Export commonly used configurations
__all__ = [
    'VECTOR_DB_TYPE', 'OLLAMA_BASE_URL', 'LLAMA_MODEL', 'LLM_TIMEOUT',
    'EMBEDDINGS_MODEL', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 'DATA_DIR',
    'SEARCH_TOP_K', 'DEVICE', 'GUI_TITLE', 'GUI_GEOMETRY',
    'SYSTEM_PROMPT', 'ERROR_MESSAGES', 'CHROMADB_COLLECTION_NAME',
    'CHROMADB_PERSIST_DIRECTORY', 'FAISS_INDEX_TYPE', 'SIMILARITY_THRESHOLD',
    'get_config_summary', 'validate_config', 'get_vector_db_config',
    'switch_vector_db', 'CHROMADB_AVAILABLE'
]