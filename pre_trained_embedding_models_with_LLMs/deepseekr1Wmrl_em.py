import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
import time
import json
from typing import List, Dict, Any, Tuple, Optional
import requests

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Groq API configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "deepseek-r1-distill-llama-70b"

# Print API key status (masked for security)
if GROQ_API_KEY:
    masked_key = GROQ_API_KEY[:4] + "*" * (len(GROQ_API_KEY) - 8) + GROQ_API_KEY[-4:]
    logging.info(f"Using Groq API key: {masked_key}")
else:
    logging.warning("No Groq API key found in environment variables")

def create_vector_db(text_folder, tables_folder, model_path, dimension=256, index_path="mrl_legal_faiss_index"):
    """
    Create a FAISS index from text files and CSV tables using MRL embeddings.
    
    Args:
        text_folder: Path to folder containing text files
        tables_folder: Path to folder containing CSV tables
        model_path: Path to MRL embedding model
        dimension: Dimension to truncate embeddings to (default: 256)
        index_path: Path to save the FAISS index
    """
    # Load MRL model
    logging.info(f"Loading MRL model from {model_path}")
    model = SentenceTransformer(model_path)
    
    # Load text documents
    documents = []
    document_sources = []
    document_types = []
    
    # Load plain text files
    logging.info(f"Processing text files from {text_folder}")
    for file in os.listdir(text_folder):
        if file.endswith('.txt'):
            file_path = os.path.join(text_folder, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # Split text into smaller chunks
                    chunks = [text[i:i+500] for i in range(0, len(text), 400)]  # 500 chars with 100 overlap
                    documents.extend(chunks)
                    document_sources.extend([file_path] * len(chunks))
                    document_types.extend(['text'] * len(chunks))
                    logging.info(f"  Processed {file}: created {len(chunks)} chunks")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
    
    # Load CSV tables
    logging.info(f"Processing CSV files from {tables_folder}")
    for file in os.listdir(tables_folder):
        if file.endswith('.csv'):
            file_path = os.path.join(tables_folder, file)
            try:
                df = pd.read_csv(file_path)
                
                # Convert table to a text representation
                table_text = f"Table: {file[:-4]}\n"
                table_text += df.to_string(index=False)
                
                documents.append(table_text)
                document_sources.append(file_path)
                document_types.append('table')
                logging.info(f"  Processed table {file}")
            except Exception as e:
                logging.error(f"Error processing {file}: {str(e)}")
    
    logging.info(f"Created {len(documents)} chunks from text files and tables")
    
    # Process in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        batch = documents[i:end_idx]
        
        logging.info(f"Encoding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        # Generate full embeddings
        batch_embeddings = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
        
        # Truncate to specified dimension for MRL
        if dimension and dimension < batch_embeddings.shape[1]:
            logging.info(f"Truncating embeddings from {batch_embeddings.shape[1]} to {dimension} dimensions")
            batch_embeddings = batch_embeddings[:, :dimension]
        
        all_embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    logging.info(f"Creating FAISS index with dimension {dimension}")
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings.astype(np.float32))
    
    # Save the index, documents, and sources
    logging.info(f"Saving index and metadata")
    index_path_with_dim = f"{index_path}_{dimension}d"
    faiss.write_index(index, f"{index_path_with_dim}.index")
    
    with open(f"{index_path_with_dim}.pkl", 'wb') as f:
        pickle.dump({
            'documents': documents, 
            'sources': document_sources,
            'types': document_types,
            'dimension': dimension
        }, f)
    
    logging.info(f"MRL vector index saved to {index_path_with_dim}.index and {index_path_with_dim}.pkl")
    
    return {
        'index': index,
        'documents': documents,
        'sources': document_sources,
        'types': document_types,
        'dimension': dimension
    }

def load_vector_db(dimension=256, index_path="mrl_legal_faiss_index"):
    """
    Load the saved MRL vector database.
    
    Args:
        dimension: Dimension of the MRL embeddings
        index_path: Base path of the FAISS index
    """
    index_path_with_dim = f"{index_path}_{dimension}d"
    
    try:
        logging.info(f"Loading existing MRL vector database ({dimension}d)...")
        index = faiss.read_index(f"{index_path_with_dim}.index")
        
        with open(f"{index_path_with_dim}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        return {
            'index': index,
            'documents': data['documents'],
            'sources': data['sources'],
            'types': data['types'],
            'dimension': data.get('dimension', dimension)
        }
    except Exception as e:
        logging.error(f"Error loading MRL vector database: {str(e)}")
        return None

def search_documents(query, model_path, db, dimension=256, k=5):
    """
    Search the MRL vector database for relevant documents.
    
    Args:
        query: The search query
        model_path: Path to MRL embedding model
        db: The loaded vector database
        dimension: Dimension to truncate embeddings to
        k: Number of results to return
    """
    # Load the model
    model = SentenceTransformer(model_path)
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Truncate to specified dimension for MRL
    if dimension and dimension < query_embedding.shape[1]:
        query_embedding = query_embedding[:, :dimension]
    
    # Search the index
    distances, indices = db['index'].search(query_embedding.astype(np.float32), k=k)
    
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
    
    return results

def generate_answer_with_groq(query, retrieved_docs, max_tokens=1024, temperature=0.1):
    """
    Generate an answer using the Groq API with DeepSeek-R1 model based on retrieved documents.
    """
    start_time = time.time()
    
    # Check if API key is available
    if not GROQ_API_KEY:
        return "Error: No Groq API key found. Please set the GROQ_API_KEY environment variable.", {
            "time": 0,
            "tokens": 0
        }
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(retrieved_docs):
        content = doc.get('content', doc.get('text', '')) # Handle different document formats
        context += f"Document {i+1}:\n{content}\n\n"
    
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
        logging.info(f"Sending request to Groq API for query: {query[:50]}...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        
        # Check for HTTP errors
        if response.status_code != 200:
            error_msg = f"Groq API error: {response.status_code} - {response.text}"
            logging.error(error_msg)
            return f"Error: {error_msg}", {
                "time": time.time() - start_time,
                "tokens": 0
            }
        
        # Extract the answer from the response
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        
        # Calculate generation time and token usage
        generation_time = time.time() - start_time
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        logging.info(f"Successfully generated answer in {generation_time:.2f} seconds")
        return answer, {
            "time": generation_time,
            "tokens": tokens_used
        }
    except Exception as e:
        error_msg = f"Error generating answer with Groq DeepSeek-R1: {str(e)}"
        logging.error(error_msg)
        return f"Error: {error_msg}", {
            "time": time.time() - start_time,
            "tokens": 0
        }

def rag_pipeline(query, model_path, dimension=256, k=5):
    """
    Complete RAG pipeline with MRL embeddings:
    1. Retrieve relevant documents using MRL embeddings
    2. Generate an answer using the Groq API with DeepSeek-R1 model
    
    Args:
        query: The user query
        model_path: Path to MRL embedding model
        dimension: Dimension to use for MRL embeddings
        k: Number of documents to retrieve
    """
    # Step 1: Load or create vector database
    db = load_vector_db(dimension=dimension)
    
    if not db:
        return "Error: Could not load or create MRL vector database.", None, None
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents for query using MRL embeddings ({dimension}d): {query}")
    search_results = search_documents(query, model_path, db, dimension=dimension, k=k)
    
    # Step 3: Generate answer
    logging.info(f"Generating answer with Groq DeepSeek-R1 ({MODEL_NAME})...")
    answer, generation_info = generate_answer_with_groq(query, search_results)
    
    logging.info(f"Answer generated with {generation_info['tokens']} tokens in {generation_info['time']:.2f} seconds")
    
    return answer, search_results, generation_info

def compare_dimensions(query, model_path, dimensions=[64, 128, 256, 512, 768], k=5):
    """
    Compare retrieval results across different MRL embedding dimensions.
    
    Args:
        query: The search query
        model_path: Path to MRL embedding model
        dimensions: List of dimensions to compare
        k: Number of results to return
    """
    results = {}
    
    for dim in dimensions:
        logging.info(f"Testing dimension: {dim}")
        
        # Load the appropriate vector database
        db = load_vector_db(dimension=dim)
        if not db:
            results[dim] = {"error": f"Could not load vector database for dimension {dim}"}
            continue
        
        # Search with this dimension
        dim_results = search_documents(query, model_path, db, dimension=dim, k=k)
        results[dim] = dim_results
    
    return results

def interactive_mode(model_path, dimension=256):
    """
    Interactive CLI for the MRL RAG system.
    """
    print(f"\n=== Sri Lankan Urban Development Regulations Assistant (Groq DeepSeek-R1 with MRL) ===")
    print(f"Using model: {MODEL_NAME}")
    print(f"Using MRL embeddings with dimension: {dimension}")
    print("Type 'exit' to quit, 'dimension X' to change dimension, 'compare' to compare dimensions\n")
    
    current_dimension = dimension
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        if query.lower().startswith('dimension '):
            try:
                new_dim = int(query.split()[1])
                if new_dim in [64, 128, 256, 512, 768]:
                    current_dimension = new_dim
                    print(f"Switched to {current_dimension}d embeddings")
                else:
                    print("Supported dimensions: 64, 128, 256, 512, 768")
            except:
                print("Invalid dimension format. Use 'dimension X' where X is a number.")
            continue
        
        if query.lower() == 'compare':
            compare_query = input("Enter your question for dimension comparison: ")
            dimension_results = compare_dimensions(
                compare_query, 
                model_path=model_path
            )
            
            for dim, results in dimension_results.items():
                if "error" in results:
                    print(f"\n=== Dimension {dim}d: ERROR ===")
                    print(results["error"])
                    continue
                    
                print(f"\n=== Dimension {dim}d Results ===")
                for i, doc in enumerate(results):
                    print(f"{i+1}. [Score: {doc['similarity']:.4f}] {doc['content'][:100]}...")
            
            continue
        
        answer, results, info = rag_pipeline(
            query,
            model_path=model_path,
            dimension=current_dimension,
            k=5
        )
        
        print(f"\n=== Retrieved Documents (MRL {current_dimension}d) ===")
        for i, doc in enumerate(results):
            print(f"{i+1}. [Score: {doc['similarity']:.4f}] Source: {os.path.basename(doc['source'])}")
            print(f"   Excerpt: {doc['content'][:150]}...\n")
        
        print("\n=== Generated Answer (DeepSeek) ===")
        print(answer)
        
        if info:
            print(f"\nGeneration time: {info['time']:.2f} seconds, Tokens used: {info['tokens']}")

def batch_process_queries(queries, model_path, dimension=256, output_file="deepseek_mrl_results.json"):
    """
    Process a batch of queries and save the results to a file.
    """
    results = {}
    
    for query in queries:
        logging.info(f"Processing query with MRL ({dimension}d): {query}")
        
        answer, retrieved_docs, info = rag_pipeline(
            query,
            model_path=model_path,
            dimension=dimension
        )
        
        results[query] = {
            "answer": answer,
            "retrieved_docs": [
                {
                    "content": doc["content"],
                    "source": doc["source"],
                    "similarity": doc["similarity"]
                }
                for doc in retrieved_docs
            ],
            "generation_info": info,
            "dimension": dimension
        }
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    # Configuration
    MRL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/data/my-mrl-embeddings" 
    MRL_DIMENSION = 256  
    
    # Create vector database if it doesn't exist
    index_path_with_dim = f"mrl_legal_faiss_index_{MRL_DIMENSION}d"
    if not os.path.exists(f"{index_path_with_dim}.index"):
        create_vector_db(
            text_folder="data/cleaned_txt",  
            tables_folder="data/table_data", 
            model_path=MRL_MODEL_PATH,
            dimension=MRL_DIMENSION
        )
    
    # Start interactive mode
    interactive_mode(MRL_MODEL_PATH, dimension=MRL_DIMENSION)
