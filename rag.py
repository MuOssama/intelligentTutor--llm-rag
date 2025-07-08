import pickle
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import uuid

import numpy as np
import faiss
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ChromaDB imports (install with: pip install chromadb)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Import configuration from your config file
from config import *

class DocumentProcessor:
    """Handles PDF processing and text extraction"""
    
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = DEVICE
        self.model = None
    
    def load_embedding_model(self, model_name: str = EMBEDDINGS_MODEL):
        """Load embedding model"""
        if self.model is None:
            self.model = SentenceTransformer(model_name, device=self.device)
        return self.model
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page info"""
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                pages.append({
                    "page_number": page_num + 1,
                    "text": text.strip()
                })
        
        doc.close()
        return pages
    
    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_text(text)
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process PDF into chunks with metadata"""
        pages = self.extract_text_from_pdf(pdf_path)
        
        full_text = "\n\n".join([p["text"] for p in pages])
        chunks = self.chunk_text(full_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "text": chunk,
                "chunk_id": i,
                "paper_title": Path(pdf_path).stem,
                "page_range": f"1-{len(pages)}",
                "total_chunks": len(chunks),
                "document_id": str(uuid.uuid4()),
                "created_at": datetime.now().isoformat()
            })
        
        return processed_chunks

class FAISSVectorDatabase:
    """FAISS-based vector database implementation"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.index = None
        self.chunks = []
        self.embeddings = None
        
        # File paths
        self.index_file = self.data_dir / "vector_db.index"
        self.chunks_file = self.data_dir / "chunks.pkl"
        self.metadata_file = self.data_dir / "metadata.json"
    
    def create_embeddings(self, chunks: List[Dict], model: SentenceTransformer):
        """Create and store embeddings"""
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index based on configuration
        dimension = embeddings.shape[1]
        
        if FAISS_INDEX_TYPE == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif FAISS_INDEX_TYPE == "IndexIVFFlat":
            nlist = min(100, len(chunks) // 10)  # Adaptive nlist
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(embeddings.astype('float32'))
        elif FAISS_INDEX_TYPE == "IndexHNSWFlat":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.chunks = chunks
        self.embeddings = embeddings
        
        self.save()
    
    def save(self):
        """Save database to disk"""
        faiss.write_index(self.index, str(self.index_file))
        
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "index_type": FAISS_INDEX_TYPE
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self) -> bool:
        """Load database from disk"""
        if not (self.index_file.exists() and self.chunks_file.exists()):
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_file))
            
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading FAISS database: {e}")
            return False
    
    def search(self, query: str, model: SentenceTransformer, top_k: int = SEARCH_TOP_K) -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None:
            return []
        
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and score >= SIMILARITY_THRESHOLD:
                chunk = self.chunks[idx].copy()
                chunk["similarity"] = float(score)
                results.append(chunk)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "type": "FAISS",
            "num_chunks": len(self.chunks),
            "index_type": FAISS_INDEX_TYPE,
            "is_trained": self.index.is_trained if self.index else False
        }

class ChromaDBVectorDatabase:
    """ChromaDB-based vector database implementation"""
    
    def __init__(self, data_dir: str = DATA_DIR):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.data_dir = Path(data_dir)
        self.persist_directory = Path(CHROMADB_PERSIST_DIRECTORY)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = None
        self.collection_name = CHROMADB_COLLECTION_NAME
        
        # Try to get existing collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = None
    
    def create_embeddings(self, chunks: List[Dict], model: SentenceTransformer):
        """Create and store embeddings"""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # Create new collection
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for ChromaDB
        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["document_id"] for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                "chunk_id": chunk["chunk_id"],
                "paper_title": chunk["paper_title"],
                "page_range": chunk["page_range"],
                "total_chunks": chunk["total_chunks"],
                "created_at": chunk["created_at"]
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks to ChromaDB collection")
    
    def load(self) -> bool:
        """Load database from disk"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            return self.collection.count() > 0
        except:
            return False
    
    def search(self, query: str, model: SentenceTransformer, top_k: int = SEARCH_TOP_K) -> List[Dict]:
        """Search for similar chunks"""
        if self.collection is None:
            return []
        
        # Generate query embedding
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents']:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                if (1 - distance) >= SIMILARITY_THRESHOLD:  # Convert distance to similarity
                    result = {
                        "text": doc,
                        "similarity": 1 - distance,
                        "chunk_id": metadata["chunk_id"],
                        "paper_title": metadata["paper_title"],
                        "page_range": metadata["page_range"],
                        "total_chunks": metadata["total_chunks"],
                        "created_at": metadata["created_at"]
                    }
                    formatted_results.append(result)
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        if self.collection is None:
            return {"type": "ChromaDB", "num_chunks": 0}
        
        return {
            "type": "ChromaDB",
            "num_chunks": self.collection.count(),
            "collection_name": self.collection_name
        }

class VectorDatabaseFactory:
    """Factory class to create vector database instances"""
    
    @staticmethod
    def create_database(db_type: str = VECTOR_DB_TYPE, data_dir: str = DATA_DIR):
        """Create vector database instance based on configuration"""
        if db_type.lower() == "faiss":
            return FAISSVectorDatabase(data_dir)
        elif db_type.lower() == "chromadb":
            if not CHROMADB_AVAILABLE:
                print("ChromaDB not available, falling back to FAISS")
                return FAISSVectorDatabase(data_dir)
            return ChromaDBVectorDatabase(data_dir)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

class RAGSystem:
    """Main RAG system orchestrator with configurable vector database"""
    
    def __init__(self, data_dir: str = DATA_DIR, db_type: str = VECTOR_DB_TYPE):
        self.processor = DocumentProcessor()
        self.vector_db = VectorDatabaseFactory.create_database(db_type, data_dir)
        self.model = None
        self.db_type = db_type
    
    def initialize(self):
        """Initialize the RAG system"""
        self.model = self.processor.load_embedding_model()
        return self.vector_db.load()
    
    def process_document(self, pdf_path: str) -> bool:
        """Process a PDF document"""
        try:
            print(f"Processing document with {self.db_type} backend...")
            
            # Process PDF
            chunks = self.processor.process_pdf(pdf_path)
            print(f"Created {len(chunks)} chunks")
            
            # Load model if not loaded
            if self.model is None:
                self.model = self.processor.load_embedding_model()
            
            # Create embeddings
            self.vector_db.create_embeddings(chunks, self.model)
            print(f"Successfully processed document using {self.db_type}")
            
            return True
        except Exception as e:
            print(f"Error processing document: {e}")
            return False
    
    def query(self, question: str, top_k: int = SEARCH_TOP_K) -> List[Dict]:
        """Query the RAG system"""
        if self.model is None:
            self.model = self.processor.load_embedding_model()
        
        results = self.vector_db.search(question, self.model, top_k)
        print(f"Found {len(results)} relevant chunks")
        return results
    
    def get_status(self) -> Dict:
        """Get system status"""
        status = {
            "database_type": self.db_type,
            "database_loaded": self.vector_db.collection is not None if hasattr(self.vector_db, 'collection') else self.vector_db.index is not None,
            "device": self.processor.device,
            "model_loaded": self.model is not None,
            "chromadb_available": CHROMADB_AVAILABLE
        }
        
        # Add database-specific stats
        status.update(self.vector_db.get_stats())
        
        return status

# Example usage
if __name__ == "__main__":
    # Test with different configurations
    
    # Test with FAISS
    print("Testing with FAISS...")
    VECTOR_DB_TYPE = "faiss"
    rag_faiss = RAGSystem()
    rag_faiss.initialize()
    
    # Test with ChromaDB (if available)
    if CHROMADB_AVAILABLE:
        print("Testing with ChromaDB...")
        VECTOR_DB_TYPE = "chromadb"
        rag_chroma = RAGSystem()
        rag_chroma.initialize()
    
    # Print status
    print("FAISS Status:", rag_faiss.get_status())
    if CHROMADB_AVAILABLE:
        print("ChromaDB Status:", rag_chroma.get_status())