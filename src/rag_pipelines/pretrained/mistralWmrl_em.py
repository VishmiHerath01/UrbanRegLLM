import os
import time
import json
import logging
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from llama_cpp import Llama


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


MRL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/data/my-mrl-embeddings"
MISTRAL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_NAME = "Mistral-7B-Instruct-v0.1"

def create_vector_db(text_folder, tables_folder, model_path, dimension=256, index_path=None):
    """
    Create a vector database using MRL embeddings with the specified dimension.
    
    Args:
        text_folder: Folder containing text documents
        tables_folder: Folder containing table documents
        model_path: Path to the MRL embedding model
        dimension: Dimension to use for MRL embeddings
        index_path: Path to save the FAISS index
    """
    if index_path is None:
        index_path = f"mrl_legal_faiss_index_{dimension}d"
    
    start_time = time.time()
    
    # Load the model
    logging.info(f"Loading MRL embedding model from {model_path}...")
    model = SentenceTransformer(model_path)
    

    embedding_dim = dimension
    logging.info(f"Using MRL dimension: {embedding_dim}")
    

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
    

    logging.info(f"Creating MRL embeddings with dimension {embedding_dim} for {len(documents)} documents...")
    
    # For MRL embeddings, we need to set the output dimension
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    # Truncate to the desired dimension
    embeddings = embeddings[:, :embedding_dim]
    

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

def load_vector_db(dimension=256, index_path=None):
    """
    Load a vector database from disk.
    
    Args:
        dimension: Dimension of the MRL embeddings
        index_path: Path to the FAISS index
    """
    if index_path is None:
        index_path = f"mrl_legal_faiss_index_{dimension}d"
    
    # Check if index exists
    if not os.path.exists(f"{index_path}.index") or not os.path.exists(f"{index_path}.pkl"):
        logging.error(f"Vector database not found at {index_path}")
        return None
    

    logging.info(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(f"{index_path}.index")
    

    with open(f"{index_path}.pkl", 'rb') as f:
        data = pickle.load(f)
    
    return {
        'index': index,
        'documents': data['documents'],
        'sources': data['sources'],
        'types': data['types']
    }

def search_documents(query, model_path, db, dimension=256, k=5):
    """
    Search for documents using MRL embeddings.
    
    Args:
        query: The search query
        model_path: Path to the MRL embedding model
        db: The vector database
        dimension: Dimension to use for MRL embeddings
        k: Number of results to return
    """
    start_time = time.time()
    
    # Load the model
    model = SentenceTransformer(model_path)
    

    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Truncate to the desired dimension
    query_embedding = query_embedding[:, :dimension]
    
    # Search the index
    distances, indices = db['index'].search(query_embedding.astype(np.float32), k)
    
    # Prepare results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(db['documents']):
            results.append({
                'content': db['documents'][idx],
                'source': db['sources'][idx],
                'type': db['types'][idx],
                'similarity': 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
            })
    
    search_time = time.time() - start_time
    logging.info(f"Search completed in {search_time:.4f} seconds")
    
    return results, search_time

def load_mistral_model():
    """Load the Mistral model for answer generation"""
    if not os.path.exists(MISTRAL_MODEL_PATH):
        logging.error(f"Mistral model not found at {MISTRAL_MODEL_PATH}")
        return None
    
    try:
        logging.info(f"Loading Mistral model from {MISTRAL_MODEL_PATH}...")
        

        model = Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_ctx=4096,  # Context window
            n_threads=8,  # CPU threads
            n_gpu_layers=0
        )
        
        logging.info("Mistral model loaded successfully")
        return model
    
    except Exception as e:
        logging.error(f"Error loading Mistral model: {str(e)}")
        return None

def generate_answer(query, context, model):
    """Generate an answer using the Mistral model"""
    # Prepare the prompt
    prompt = f"""<s>[INST] You are a helpful assistant that answers questions about Sri Lankan Urban Development Regulations. 
Use ONLY the following information to answer the question. If you don't know the answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query} [/INST]
"""
    
    try:

        response = model.create_completion(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1,
            top_k=40,
            echo=False
        )
        
        # Extract the answer and token count
        answer = response['choices'][0]['text'].strip()
        tokens = response['usage']['completion_tokens']
        
        return answer, tokens
    
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return f"Error: {str(e)}", 0

def rag_pipeline(query, model_path=MRL_MODEL_PATH, dimension=256, k=5):
    """
    Complete RAG pipeline with MRL embeddings:
    1. Retrieve relevant documents using MRL embeddings
    2. Generate an answer using the Mistral model
    
    Args:
        query: The user query
        model_path: Path to MRL embedding model
        dimension: Dimension to use for MRL embeddings
        k: Number of documents to retrieve
    """
    # Step 1: Load or create vector database
    index_path = f"mrl_legal_faiss_index_{dimension}d"
    db = load_vector_db(dimension, index_path)
    
    if not db:
        logging.info(f"Vector database not found. Creating new one with dimension {dimension}...")
        cleaned_text_dir = os.path.join(DATA_DIR, "cleaned_txt")
        tables_dir = os.path.join(DATA_DIR, "table_data")
        
        if not os.path.exists(cleaned_text_dir) or not os.path.exists(tables_dir):
            return "Error: Data directories not found.", None, None
        
        db = create_vector_db(cleaned_text_dir, tables_dir, model_path, dimension, index_path)
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents for query: {query}")
    search_results, search_time = search_documents(query, model_path, db, dimension, k=k)
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(search_results):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    
    # Step 3: Load Mistral model
    model = load_mistral_model()
    if not model:
        return "Error: Could not load Mistral model.", search_results, None
    
    # Step 4: Generate answer
    logging.info("Generating answer with Mistral...")
    answer, tokens = generate_answer(query, context, model)
    
    logging.info(f"Answer generated with {tokens} tokens")
    
    # Return the answer, retrieved documents, and metrics
    return answer, search_results, {
        "search_time": search_time,
        "tokens": tokens,
        "dimension": dimension
    }

def interactive_mode(model_path=MRL_MODEL_PATH, dimension=256):
    """
    Interactive CLI for the RAG system.
    
    Args:
        model_path: Path to the MRL embedding model
        dimension: Dimension to use for MRL embeddings
    """
    print(f"\n=== Sri Lankan Urban Development Regulations Assistant (Mistral with MRL Embeddings {dimension}d) ===")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        print("\nProcessing your query...")
        answer, docs, metrics = rag_pipeline(query, model_path, dimension)
        
        print("\n=== Answer ===")
        print(answer)
        
        if metrics:
            print(f"\nSearch time: {metrics['search_time']:.4f}s, Tokens used: {metrics['tokens']}")
        
        # Option to show retrieved documents
        show_docs = input("\nShow retrieved documents? (y/n): ")
        if show_docs.lower() == 'y':
            print("\n=== Retrieved Documents ===")
            for i, doc in enumerate(docs):
                print(f"\nDocument {i+1} ({doc['source']}, similarity: {doc['similarity']:.4f}):")
                print("-" * 50)
                print(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                print("-" * 50)

def batch_mode(queries_file, model_path=MRL_MODEL_PATH, dimension=256):
    """
    Process a batch of queries from a file.
    
    Args:
        queries_file: Path to a JSON file containing queries
        model_path: Path to the MRL embedding model
        dimension: Dimension to use for MRL embeddings
    """
    try:
        with open(queries_file, 'r') as f:
            queries = json.load(f)
        
        results = []
        
        for query_item in queries:
            query = query_item['question']
            logging.info(f"Processing query: {query}")
            
            answer, docs, metrics = rag_pipeline(query, model_path, dimension)
            
            results.append({
                'query': query,
                'answer': answer,
                'metrics': metrics
            })
        
        # Save results
        output_file = os.path.join(RESULTS_DIR, f'mrl_{dimension}d_batch_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Batch processing complete. Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error in batch processing: {str(e)}")

if __name__ == "__main__":

    dimension = 256
    

    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            try:
                dimension = int(sys.argv[2])
            except ValueError:
                pass
        batch_mode(sys.argv[1], dimension=dimension)
    else:

        interactive_mode(dimension=dimension)
