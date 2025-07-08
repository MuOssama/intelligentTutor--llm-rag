import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from typing import Optional, Callable
from config import *

class SimpleGUI:
    def __init__(self):
        self.root = tk.Tk()  # Initialize tkinter root
        self.root.title(GUI_TITLE)
        self.root.geometry(GUI_GEOMETRY)
        
        # Callbacks - to be set by main application
        self.on_process_pdf: Optional[Callable] = None
        self.on_query: Optional[Callable] = None
        self.on_check_llm: Optional[Callable] = None
        self.on_pull_model: Optional[Callable] = None
        
        # Variables
        self.file_path = tk.StringVar()
        self.status_message = tk.StringVar(value="Ready")
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Setup tab
        setup_frame = ttk.Frame(notebook)
        notebook.add(setup_frame, text="Setup")
        self.create_setup_tab(setup_frame)
        
        # Query tab
        query_frame = ttk.Frame(notebook)
        notebook.add(query_frame, text="Query")
        self.create_query_tab(query_frame)
        
        # Status bar
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Label(status_bar, textvariable=self.status_message).pack(side=tk.LEFT, padx=5)
    
    def create_setup_tab(self, parent):
        """Create setup tab"""
        # LLM Status
        llm_frame = ttk.LabelFrame(parent, text="LLM Status", padding=10)
        llm_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.llm_status = ttk.Label(llm_frame, text="LLM: Not checked")
        self.llm_status.pack(side=tk.LEFT)
        
        ttk.Button(llm_frame, text="Check LLM", command=self.check_llm).pack(side=tk.RIGHT, padx=5)
        ttk.Button(llm_frame, text="Pull Model", command=self.pull_model).pack(side=tk.RIGHT)
        
        # PDF Processing
        pdf_frame = ttk.LabelFrame(parent, text="PDF Processing", padding=10)
        pdf_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # File selection
        file_frame = ttk.Frame(pdf_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="PDF:").pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.RIGHT)
        
        # Process button
        ttk.Button(pdf_frame, text="Process PDF", command=self.process_pdf).pack(pady=10)
        
        # Progress
        self.progress = ttk.Progressbar(pdf_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Log
        log_frame = ttk.LabelFrame(pdf_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_query_tab(self, parent):
        """Create query tab"""
        # Query input
        input_frame = ttk.LabelFrame(parent, text="Ask Question", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(input_frame, text="Question:").pack(anchor=tk.W)
        
        self.query_text = scrolledtext.ScrolledText(input_frame, height=3)
        self.query_text.pack(fill=tk.X, pady=5)
        
        ttk.Button(input_frame, text="Get Answer", command=self.query_system).pack(pady=5)
        
        # Results
        results_frame = ttk.LabelFrame(parent, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame)
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """Browse for PDF file"""
        filename = filedialog.askopenfilename(
            title="Select PDF",
            filetypes=[("PDF files", "*.pdf")]
        )
        if filename:
            self.file_path.set(filename)
    
    def process_pdf(self):
        """Process PDF - calls callback"""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a PDF file")
            return
        
        if self.on_process_pdf:
            self.progress.start()
            threading.Thread(target=self._process_pdf_thread, daemon=True).start()
    
    def _process_pdf_thread(self):
        """Process PDF in thread"""
        try:
            success = self.on_process_pdf(self.file_path.get())
            self.progress.stop()
            
            if success:
                self.log("PDF processed successfully!")
                messagebox.showinfo("Success", "PDF processed successfully!")
            else:
                self.log("Failed to process PDF")
                messagebox.showerror("Error", "Failed to process PDF")
        except Exception as e:
            self.progress.stop()
            self.log(f"Error: {e}")
            messagebox.showerror("Error", str(e))
    
    def query_system(self):
        """Query system - calls callback"""
        query = self.query_text.get(1.0, tk.END).strip()
        if not query:
            messagebox.showwarning("Warning", "Please enter a question")
            return
        
        if self.on_query:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Searching...\n")
            threading.Thread(target=self._query_thread, args=(query,), daemon=True).start()
    
    def _query_thread(self, query: str):
        """Query in thread"""
        try:
            result = self.on_query(query)
            self.display_results(query, result)
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {e}")
    
    def check_llm(self):
        """Check LLM - calls callback"""
        if self.on_check_llm:
            threading.Thread(target=self._check_llm_thread, daemon=True).start()
    
    def _check_llm_thread(self):
        """Check LLM in thread"""
        try:
            status = self.on_check_llm()
            if status:
                self.llm_status.config(text="LLM: Ready")
                self.log("LLM is ready")
            else:
                self.llm_status.config(text="LLM: Not ready")
                self.log("LLM not ready")
        except Exception as e:
            self.llm_status.config(text="LLM: Error")
            self.log(f"LLM check error: {e}")
    
    def pull_model(self):
        """Pull model - calls callback"""
        if self.on_pull_model:
            threading.Thread(target=self._pull_model_thread, daemon=True).start()
    
    def _pull_model_thread(self):
        """Pull model in thread"""
        try:
            self.log("Pulling model...")
            success = self.on_pull_model()
            if success:
                self.log("Model pulled successfully")
            else:
                self.log("Failed to pull model")
        except Exception as e:
            self.log(f"Pull model error: {e}")
    
    def display_results(self, query: str, result: dict):
        """Display query results"""
        self.results_text.delete(1.0, tk.END)
        
        self.results_text.insert(tk.END, f"QUESTION: {query}\n\n")
        
        if result.get("error"):
            self.results_text.insert(tk.END, f"ERROR: {result['error']}\n")
            return
        
        # Answer
        self.results_text.insert(tk.END, "ANSWER:\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n")
        self.results_text.insert(tk.END, f"{result.get('answer', 'No answer')}\n\n")
        
        # Sources
        sources = result.get("sources", [])
        if sources:
            self.results_text.insert(tk.END, "SOURCES:\n")
            self.results_text.insert(tk.END, "=" * 50 + "\n")
            
            for i, source in enumerate(sources, 1):
                self.results_text.insert(tk.END, f"\n{i}. {source.get('paper_title', 'Unknown')}\n")
                self.results_text.insert(tk.END, f"   Similarity: {source.get('similarity', 0):.3f}\n")
                self.results_text.insert(tk.END, f"   Pages: {source.get('page_range', 'Unknown')}\n")
                
                # Preview
                text = source.get('text', '')
                preview = text[:200] + "..." if len(text) > 200 else text
                self.results_text.insert(tk.END, f"   Preview: {preview}\n")
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def set_status(self, message: str):
        """Set status message"""
        self.status_message.set(message)
        self.root.update()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()
    
    def close(self):
        """Close the GUI"""
        self.root.destroy()

# Example usage
if __name__ == "__main__":
    gui = SimpleGUI()
    gui.run()