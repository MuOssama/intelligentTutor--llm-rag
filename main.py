#!/usr/bin/env python3
"""
Main application entry point
Simple RAG application with modular architecture
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import *

from llm import OllamaLLM
from rag import RAGSystem
from gui import SimpleGUI

class RAGApplication:
    """Main RAG Application orchestrator"""
    
    def __init__(self):
        # Validate configuration
        config_errors = validate_config()
        if config_errors:
            print("Configuration errors found:")
            for error in config_errors:
                print(f"  - {error}")
            print("Using default values where needed.\n")
        
        # Initialize components with config values
        self.llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=LLAMA_MODEL)
        self.rag = RAGSystem(data_dir=DATA_DIR)
        self.gui = SimpleGUI()  # Remove the parameter that doesn't exist
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Initialize
        self.initialize()
    
    def setup_callbacks(self):
        """Setup GUI callbacks"""
        self.gui.on_process_pdf = self.process_pdf
        self.gui.on_query = self.query
        self.gui.on_check_llm = self.check_llm
        self.gui.on_pull_model = self.pull_model
    
    def initialize(self):
        """Initialize the application"""
        self.gui.log("Initializing RAG system...")
        
        # Log configuration summary
        config_summary = get_config_summary()
        for key, value in config_summary.items():
            self.gui.log(f"{key}: {value}")
        
        # Load existing database if available
        if self.rag.initialize():
            self.gui.log("Database loaded successfully")
            status = self.rag.get_status()
            self.gui.log(f"Loaded {status['num_chunks']} chunks")
        else:
            self.gui.log("No existing database found")
        
        # Check LLM
        if self.llm.is_available():
            self.gui.llm_status.config(text="LLM: Ready")
            self.gui.log("LLM is ready")
        else:
            self.gui.llm_status.config(text="LLM: Not ready")
            self.gui.log("LLM not ready - check Ollama setup")
    
    def process_pdf(self, pdf_path: str) -> bool:
        """Process PDF callback"""
        self.gui.log(f"Processing PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            self.gui.log("PDF file not found")
            return False
        
        success = self.rag.process_document(pdf_path)
        
        if success:
            status = self.rag.get_status()
            self.gui.log(f"PDF processed - {status['num_chunks']} chunks created")
        else:
            self.gui.log("Failed to process PDF")
        
        return success
    
    def query(self, question: str) -> dict:
        """Query callback"""
        self.gui.log(f"Querying: {question}")
        
        # Check if system is ready
        if not self.rag.get_status()['database_loaded']:
            return {"error": "No database loaded. Please process a PDF first."}
        
        if not self.llm.is_available():
            return {"error": "LLM not available. Please check Ollama setup."}
        
        # Get relevant chunks
        chunks = self.rag.query(question, top_k=5)
        
        if not chunks:
            return {"error": "No relevant information found"}
        
        # Format context
        context = self.llm.format_context(chunks)
        
        # Generate response
        llm_response = self.llm.generate_response(question, context)
        
        if llm_response.get("error"):
            return {"error": llm_response["error"]}
        
        return {
            "answer": llm_response["response"],
            "sources": chunks,
            "model": llm_response.get("model", "unknown")
        }
    
    def check_llm(self) -> bool:
        """Check LLM callback"""
        return self.llm.is_available()
    
    def pull_model(self) -> bool:
        """Pull model callback"""
        return self.llm.pull_model()
    
    def run(self):
        """Run the application"""
        self.gui.run()

def main():
    """Main entry point"""
    try:
        # Check dependencies
        import torch
        import faiss
        import fitz
        import sentence_transformers
        
        print("All dependencies found.")
        print("Configuration summary:")
        config_summary = get_config_summary()
        for key, value in config_summary.items():
            print(f"  {key}: {value}")
        print()
        
        # Create and run app
        app = RAGApplication()
        app.run()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("pip install torch sentence-transformers faiss-cpu PyMuPDF requests numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()