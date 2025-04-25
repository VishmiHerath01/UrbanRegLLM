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


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "deepseek-r1-distill-llama-70b"


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


MRL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/data/my-mrl-embeddings"

def create_vector_db(text_folder, tables_folder, model_path, dimension=256, index_path="mrl_legal_faiss_index"):
    start_time = time.time()
    
    # Load the model
    logging.info(f"Loading MRL embedding model from {model_path}...")
    model = SentenceTransformer(model_path)
    

    embedding_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Model embedding dimension: {embedding_dim}")
    

    documents = []
    sources = []
    types = []
    

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
    

    logging.info(f"Creating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    # Truncate to specified dimension for MRL
    if dimension and dimension < embeddings.shape[1]:
        logging.info(f"Truncating embeddings from {embeddings.shape[1]}d to {dimension}d")
        embeddings = embeddings[:, :dimension]
    

    logging.info(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    # Save the index and metadata
    index_path_with_dim = f"{index_path}_{dimension}d"
    logging.info(f"Saving index to {index_path_with_dim}...")
    faiss.write_index(index, f"{index_path_with_dim}.index")
    
    # Save document data
    with open(f"{index_path_with_dim}.pkl", 'wb') as f:
        pickle.dump({
            'documents': documents,
            'sources': sources,
            'types': types,
            'dimension': dimension
        }, f)
    
    elapsed_time = time.time() - start_time
    logging.info(f"MRL vector database created in {elapsed_time:.2f} seconds")
    
    return {
        'index': index,
        'documents': documents,
        'sources': sources,
        'types': types,
        'dimension': dimension
    }

def load_vector_db(dimension=256, index_path="mrl_legal_faiss_index"):
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
    # Load the model
    model = SentenceTransformer(model_path)
    
    # Encode the query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Truncate to specified dimension for MRL
    if dimension and dimension < query_embedding.shape[1]:
        query_embedding = query_embedding[:, :dimension]
    
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
    start_time = time.time()
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    

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

def rag_pipeline(query, model_path=MRL_MODEL_PATH, dimension=256, k=5):
    # Step 1: Load or create vector database
    db = load_vector_db(dimension=dimension)
    
    if not db:
        logging.info(f"MRL vector database not found. Creating new one with dimension {dimension}...")
        cleaned_text_dir = os.path.join(DATA_DIR, "cleaned_txt")
        tables_dir = os.path.join(DATA_DIR, "table_data")
        
        if not os.path.exists(cleaned_text_dir) or not os.path.exists(tables_dir):
            return "Error: Data directories not found.", None, None
        
        db = create_vector_db(cleaned_text_dir, tables_dir, model_path, dimension=dimension)
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents for query using MRL embeddings ({dimension}d): {query}")
    search_results, search_time = search_documents(query, model_path, db, dimension=dimension, k=k)
    
    # Step 3: Generate answer
    logging.info(f"Generating answer with Groq DeepSeek-R1 ({MODEL_NAME})...")
    answer, generation_info = generate_answer_with_groq(query, search_results)
    
    logging.info(f"Answer generated with {generation_info['tokens']} tokens in {generation_info['time']:.2f} seconds")
    
    # Return the answer, retrieved documents, and metrics
    return answer, search_results, {
        "search_time": search_time,
        "generation_time": generation_info["time"],
        "tokens": generation_info["tokens"],
        "dimension": dimension
    }

def compare_dimensions(query, model_path=MRL_MODEL_PATH, k=5):
    dimensions = [64, 128, 256, 512, 768]
    results = {}
    
    for dim in dimensions:
        logging.info(f"Testing MRL dimension {dim}d...")
        answer, docs, metrics = rag_pipeline(query, model_path, dimension=dim, k=k)
        
        results[dim] = {
            "answer": answer,
            "metrics": metrics,
            "top_doc_preview": docs[0]["content"][:100] + "..." if docs else "No documents retrieved"
        }
    
    return results

def interactive_mode(model_path=MRL_MODEL_PATH, dimension=256):
    print(f"\n=== Sri Lankan Urban Development Regulations Assistant (Groq DeepSeek-R1 with MRL) ===")
    print(f"Using model: {MODEL_NAME}")
    print(f"Using MRL embeddings with dimension: {dimension}")
    print("Type 'exit' to quit, 'dimension X' to change dimension, 'compare' to compare dimensions\n")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        elif query.lower().startswith('dimension '):
            try:
                new_dim = int(query.split()[1])
                if new_dim in [64, 128, 256, 512, 768]:
                    dimension = new_dim
                    print(f"\nChanged MRL dimension to {dimension}d")
                else:
                    print("\nSupported dimensions: 64, 128, 256, 512, 768")
                continue
            except:
                print("\nInvalid dimension format. Use 'dimension X' where X is a number.")
                continue
        elif query.lower() == 'compare':
            print("\nComparing different MRL dimensions...")
            comparison = compare_dimensions(input("\nEnter query for dimension comparison: "), model_path)
            
            print("\n=== Dimension Comparison ===")
            for dim, result in comparison.items():
                print(f"\n--- {dim}d Embeddings ---")
                print(f"Search time: {result['metrics']['search_time']:.6f}s")
                print(f"Generation time: {result['metrics']['generation_time']:.2f}s")
                print(f"Tokens: {result['metrics']['tokens']}")
                print(f"Top document preview: {result['top_doc_preview']}")
            
            continue
        
        print(f"\nProcessing your query with MRL {dimension}d embeddings...")
        answer, docs, metrics = rag_pipeline(query, model_path, dimension=dimension)
        
        print("\n=== Answer ===")
        print(answer)
        
        if metrics:
            print(f"\nSearch time: {metrics['search_time']:.6f}s, Generation time: {metrics['generation_time']:.2f}s, Tokens: {metrics['tokens']}")
        
        # Option to show retrieved documents
        show_docs = input("\nShow retrieved documents? (y/n): ")
        if show_docs.lower() == 'y':
            print("\n=== Retrieved Documents ===")
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1} ({doc['source']}, similarity: {doc['similarity']:.4f}):")
                print("-" * 50)
                print(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                print("-" * 50)

def batch_mode(queries_file, dimension=256):
    try:
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        
        results = []
        
        for query_item in queries:
            query = query_item['question']
            logging.info(f"Processing query with MRL {dimension}d: {query}")
            
            answer, docs, metrics = rag_pipeline(query, dimension=dimension)
            
            results.append({
                'query': query,
                'answer': answer,
                'metrics': metrics
            })
        
        # Save results
        output_file = os.path.join(RESULTS_DIR, f'batch_results_mrl_{dimension}d.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Batch processing complete. Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")

if __name__ == "__main__":

    if len(sys.argv) > 1:
        if sys.argv[1].startswith("--dimension="):

            try:
                dimension = int(sys.argv[1].split("=")[1])
                if dimension not in [64, 128, 256, 512, 768]:
                    print("Supported dimensions: 64, 128, 256, 512, 768")
                    dimension = 256  # Default
            except:
                dimension = 256  # Default
                
            if len(sys.argv) > 2:

                batch_mode(sys.argv[2], dimension=dimension)
            else:

                interactive_mode(dimension=dimension)
        else:

            batch_mode(sys.argv[1])
    else:

        interactive_mode()
