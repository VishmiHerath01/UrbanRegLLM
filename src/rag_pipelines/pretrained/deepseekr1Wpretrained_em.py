import os
import time
import json
import logging
import sys
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Groq API configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "deepseek-r1-distill-llama-70b"

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model paths
EMBEDDING_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/legal-embeddings-model"

def create_vector_db(text_folder, tables_folder, embedding_model_path, index_path="legal_faiss_index"):
    """
    Create a vector database using the specified embedding model.
    
    Args:
        text_folder: Folder containing text documents
        tables_folder: Folder containing table documents
        embedding_model_path: Path to the embedding model
        index_path: Path to save the FAISS index
    """
    start_time = time.time()
    
    # Load the model
    logging.info(f"Loading embedding model from {embedding_model_path}...")
    model = SentenceTransformer(embedding_model_path)
    
    # Get the embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Model embedding dimension: {embedding_dim}")
    
    # Process text documents
    documents = []
    sources = []
    types = []
    
    # Process text documents
    logging.info(f"Processing text documents from {text_folder}...")
    for filename in os.listdir(text_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(text_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split the document into chunks (simple approach)
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                for chunk in chunks:
                    documents.append(chunk)
                    sources.append(filename)
                    types.append('text')
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    
    # Process table documents
    logging.info(f"Processing table documents from {tables_folder}...")
    for filename in os.listdir(tables_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(tables_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                documents.append(content)
                sources.append(filename)
                types.append('table')
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
    
    # Create embeddings
    logging.info(f"Creating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    # Create FAISS index
    logging.info(f"Creating FAISS index with dimension {embedding_dim}...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype(np.float32))
    
    # Save the index and metadata
    logging.info(f"Saving index to {index_path}...")
    faiss.write_index(index, f"{index_path}.index")
    
    # Save document data
    with open(f"{index_path}.pkl", 'wb') as f:
        pickle.dump({
            'documents': documents,
            'sources': sources,
            'types': types
        }, f)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Vector database created in {elapsed_time:.2f} seconds")
    
    return {
        'index': index,
        'documents': documents,
        'sources': sources,
        'types': types
    }

def load_vector_db(index_path="legal_faiss_index"):
    """
    Load a vector database from disk.
    
    Args:
        index_path: Path to the FAISS index
    """
    try:
        logging.info(f"Loading index from {index_path}...")
        index = faiss.read_index(f"{index_path}.index")
        
        with open(f"{index_path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        return {
            'index': index,
            'documents': data['documents'],
            'sources': data['sources'],
            'types': data['types']
        }
    except Exception as e:
        logging.error(f"Error loading vector database: {str(e)}")
        return None

def search_documents(query, embedding_model_path, db, k=5):
    """
    Search for documents using the specified model.
    
    Args:
        query: The search query
        embedding_model_path: Path to the embedding model
        db: The vector database
        k: Number of results to return
    """
    # Load the model
    model = SentenceTransformer(embedding_model_path)
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search the index
    start_time = time.time()
    distances, indices = db['index'].search(query_embedding.astype(np.float32), k=k)
    search_time = time.time() - start_time
    
    # Format the results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(db['documents']):
            results.append({
                'content': db['documents'][idx],
                'source': db['sources'][idx],
                'type': db['types'][idx],
                'similarity': float(1 - distances[0][i])  # Convert distance to similarity score
            })
    
    return results, search_time

def generate_answer_with_groq(query, retrieved_docs, max_tokens=1024, temperature=0.1):
    """
    Generate an answer using the Groq API with DeepSeek-R1 model based on retrieved documents.
    """
    start_time = time.time()
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    
    # Create the prompt
    system_message = "You are a legal assistant specialized in Sri Lankan building regulations and urban development. Answer the question based ONLY on the provided context. If the context doesn't contain relevant information, say 'I don't have enough information to answer this question.'"
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        # Make the API request
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract the answer from the response
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        # Calculate generation time and token usage
        generation_time = time.time() - start_time
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        return answer, {
            "time": generation_time,
            "tokens": tokens_used
        }
    except Exception as e:
        logging.error(f"Error generating answer with Groq DeepSeek-R1: {str(e)}")
        return f"Error generating answer: {str(e)}", {
            "time": time.time() - start_time,
            "tokens": 0
        }

def rag_pipeline(query, embedding_model_path=EMBEDDING_MODEL_PATH, k=5):
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents
    2. Generate an answer using the DeepSeek-R1 model via Groq API
    
    Args:
        query: The user query
        embedding_model_path: Path to embedding model
        k: Number of documents to retrieve
    """
    # Step 1: Load or create vector database
    db = load_vector_db()
    
    if not db:
        logging.info("Vector database not found. Creating new one...")
        cleaned_text_dir = os.path.join(DATA_DIR, "cleaned_txt")
        tables_dir = os.path.join(DATA_DIR, "table_data")
        
        if not os.path.exists(cleaned_text_dir) or not os.path.exists(tables_dir):
            return "Error: Data directories not found.", None, None
        
        db = create_vector_db(cleaned_text_dir, tables_dir, embedding_model_path)
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents for query: {query}")
    search_results, search_time = search_documents(query, embedding_model_path, db, k=k)
    
    # Step 3: Generate answer
    logging.info(f"Generating answer with Groq DeepSeek-R1 ({MODEL_NAME})...")
    answer, generation_info = generate_answer_with_groq(query, search_results)
    
    logging.info(f"Answer generated with {generation_info['tokens']} tokens in {generation_info['time']:.2f} seconds")
    
    # Return the answer, retrieved documents, and metrics
    return answer, search_results, {
        "search_time": search_time,
        "generation_time": generation_info["time"],
        "tokens": generation_info["tokens"]
    }

def interactive_mode():
    """
    Interactive CLI for the RAG system.
    """
    print(f"\n=== Sri Lankan Urban Development Regulations Assistant (Groq DeepSeek-R1) ===")
    print(f"Using model: {MODEL_NAME}")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        print("\nProcessing your query...")
        answer, docs, metrics = rag_pipeline(query)
        
        print("\n=== Answer ===")
        print(answer)
        
        if metrics:
            print(f"\nSearch time: {metrics['search_time']:.4f}s, Generation time: {metrics['generation_time']:.2f}s, Tokens used: {metrics['tokens']}")
        
        # Option to show retrieved documents
        show_docs = input("\nShow retrieved documents? (y/n): ")
        if show_docs.lower() == 'y':
            print("\n=== Retrieved Documents ===")
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1} ({doc['source']}, similarity: {doc['similarity']:.4f}):")
                print("-" * 50)
                print(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                print("-" * 50)

def batch_mode(queries_file):
    """
    Process a batch of queries from a file.
    
    Args:
        queries_file: Path to a JSON file containing queries
    """
    try:
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        
        results = []
        
        for query_item in queries:
            query = query_item['question']
            logging.info(f"Processing query: {query}")
            
            answer, docs, metrics = rag_pipeline(query)
            
            results.append({
                'query': query,
                'answer': answer,
                'metrics': metrics
            })
        
        # Save results
        output_file = os.path.join(RESULTS_DIR, 'batch_results_deepseek.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Batch processing complete. Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")

if __name__ == "__main__":
    # Check if we should run in batch mode
    if len(sys.argv) > 1:
        batch_mode(sys.argv[1])
    else:
        # Run in interactive mode
        interactive_mode()
