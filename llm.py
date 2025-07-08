import requests
import json
from typing import Dict, List, Optional
from config import *

class LLMInterface:
    """Base class for LLM implementations"""
    
    def generate_response(self, prompt: str, context: str = "") -> Dict:
        raise NotImplementedError

class OllamaLLM(LLMInterface):
    """Ollama LLM implementation for Llama 3.2:3B"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url
        self.model = model
        self.timeout = 240
        
    def is_available(self) -> bool:
        """Check if Ollama is running and has the model"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=30)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == self.model for model in models)
            return False
        except:
            return False
    
    def pull_model(self) -> bool:
        """Pull the model if not available"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300
            )
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, context: str = "") -> Dict:
        """Generate response using Ollama"""
        if not self.is_available():
            return {"error": "Ollama not available", "response": ""}
        
        system_prompt = """You are an expert computer architecture researcher. 
        Analyze research papers and provide precise, technical answers.
        Always cite sources [Source X] for your claims."""
        
        full_prompt = f"System: {system_prompt}\n\nContext: {context}\n\nUser: {prompt}"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 1024,
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", "").strip(),
                    "model": self.model,
                    "error": None
                }
            else:
                return {
                    "error": f"HTTP {response.status_code}",
                    "response": ""
                }
        except Exception as e:
            return {
                "error": str(e),
                "response": ""
            }
    
    def format_context(self, chunks: List[Dict]) -> str:
        """Format context from RAG chunks"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_part = f"""
[Source {i+1}]
Paper: {chunk.get('paper_title', 'Unknown')}
Pages: {chunk.get('page_range', 'Unknown')}
Content: {chunk.get('text', '')}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)