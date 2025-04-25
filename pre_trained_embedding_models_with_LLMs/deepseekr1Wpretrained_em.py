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
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "deepseek-r1-distill-llama-70b"

def create_vector_db(text_folder, tables_folder, model_path, index_path="legal_faiss_index"):
    """
    Create a FAISS index from text files and CSV tables using your fine-tuned model.
    """
    # Load your fine-tuned model
    logging.info(f"Loading model from {model_path}")
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
        batch_embeddings = model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
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
    faiss.write_index(index, f"{index_path}.index")
    
    with open(f"{index_path}.pkl", 'wb') as f:
        pickle.dump({
            'documents': documents, 
            'sources': document_sources,
            'types': document_types
        }, f)
    
    logging.info(f"Vector index saved to {index_path}.index and {index_path}.pkl")
    
    return {
        'index': index,
        'documents': documents,
        'sources': document_sources,
        'types': document_types
    }

def load_vector_db(index_path="legal_faiss_index"):
    """
    Load the saved vector database.
    """
    try:
        logging.info("Loading existing vector database...")
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

def search_documents(query, model_path, db, k=5):
    """
    Search the vector database for relevant documents.
    """
    # Load the model
    model = SentenceTransformer(model_path)
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
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

def rag_pipeline(query, embedding_model_path, k=5):
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents using your custom embedding model
    2. Generate an answer using the Groq API with DeepSeek-R1 model
    """
    # Step 1: Load or create vector database
    db = load_vector_db()
    
    if not db:
        return "Error: Could not load or create vector database.", None, None
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents for query: {query}")
    search_results = search_documents(query, embedding_model_path, db, k=k)
    
    # Step 3: Generate answer
    logging.info(f"Generating answer with Groq DeepSeek-R1 ({MODEL_NAME})...")
    answer, generation_info = generate_answer_with_groq(query, search_results)
    
    logging.info(f"Answer generated with {generation_info['tokens']} tokens in {generation_info['time']:.2f} seconds")
    
    return answer, search_results, generation_info

def compare_embeddings(query, custom_model_path, baseline_model_path="sentence-transformers/all-MiniLM-L6-v2", index_path="legal_faiss_index", k=5):
    """
    Compare results from your custom embedding model with a baseline model.
    """
    # Load vector database
    db = load_vector_db(index_path)
    if not db:
        return None, None
    
    # Search with custom model
    custom_results = search_documents(query, custom_model_path, db, k=k)
    
    # Search with baseline model
    baseline_results = search_documents(query, baseline_model_path, db, k=k)
    
    return custom_results, baseline_results

def interactive_mode(embedding_model_path, index_path="legal_faiss_index"):
    """
    Interactive CLI for the RAG system.
    """
    print(f"\n=== Sri Lankan Urban Development Regulations Assistant (Groq DeepSeek-R1) ===")
    print(f"Using model: {MODEL_NAME}")
    print("Type 'exit' to quit, 'compare' to compare with baseline embeddings\n")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        if query.lower() == 'compare':
            query = input("Enter your question for comparison: ")
            custom_results, baseline_results = compare_embeddings(
                query, 
                custom_model_path=embedding_model_path
            )
            
            if custom_results and baseline_results:
                print("\n=== Custom Model Results ===")
                for i, doc in enumerate(custom_results):
                    print(f"{i+1}. [Score: {doc['similarity']:.4f}] {doc['content'][:100]}...")
                
                print("\n=== Baseline Model Results ===")
                for i, doc in enumerate(baseline_results):
                    print(f"{i+1}. [Score: {doc['similarity']:.4f}] {doc['content'][:100]}...")
            
            continue
        
        answer, results, info = rag_pipeline(
            query,
            embedding_model_path=embedding_model_path,
            k=5
        )
        
        print("\n=== Retrieved Documents ===")
        for i, doc in enumerate(results):
            print(f"{i+1}. [Score: {doc['similarity']:.4f}] Source: {os.path.basename(doc['source'])}")
            print(f"   Excerpt: {doc['content'][:150]}...\n")
        
        print("\n=== Generated Answer ===")
        print(answer)
        
        if info:
            print(f"\nGeneration time: {info['time']:.2f} seconds, Tokens used: {info['tokens']}")

def batch_process_queries(queries, embedding_model_path, output_file="deepseek_results.json"):
    """
    Process a batch of queries and save the results to a file.
    """
    results = {}
    
    for query in queries:
        logging.info(f"Processing query: {query}")
        
        answer, retrieved_docs, info = rag_pipeline(
            query,
            embedding_model_path=embedding_model_path
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
            "generation_info": info
        }
    
    # Save results to file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    # Configuration
    EMBEDDING_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/legal-embeddings-model"  # Your fine-tuned embedding model
    
    # Create vector database if it doesn't exist
    if not os.path.exists("legal_faiss_index.index"):
        create_vector_db(
            text_folder="data/cleaned_txt",  # Update with your folder path
            tables_folder="data/table_data",  # Update with your folder path
            model_path=EMBEDDING_MODEL_PATH
        )
    
    # Start interactive mode
    interactive_mode(EMBEDDING_MODEL_PATH)
