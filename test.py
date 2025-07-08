from rag import RAGSystem  # Assuming your main code is saved as rag.py

# === Step 1: Initialize the RAG system ===
rag = RAGSystem(data_dir="rag_data")
db_loaded = rag.initialize()

print(f"Database Loaded: {db_loaded}")

# === Step 2: Process a test PDF ===
pdf_path = "MutspahaOssama27Jun25.pdf"  # Change this to your test file
processed = rag.process_document(pdf_path)
print(f"PDF processed: {processed}")

# === Step 3: Query the system ===
question = "did mustapah worked as instructor?"
results = rag.query(question, top_k=3)

print("\nTop Matches:")
for i, res in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Chunk ID: {res['chunk_id']}")
    print(f"Similarity: {res['similarity']:.4f}")
    print(f"Text Snippet:\n{res['text'][:300]}...")
