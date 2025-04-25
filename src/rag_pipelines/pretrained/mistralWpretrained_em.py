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

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model paths
EMBEDDING_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/legal-embeddings-model"
MISTRAL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

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

def load_mistral_model():
    """Load the Mistral model for answer generation"""
    logging.info("Loading Mistral model...")
    
    try:
        # Check if we have the quantized GGUF model
        if os.path.exists(MISTRAL_MODEL_PATH):
            logging.info(f"Using quantized GGUF model: {MISTRAL_MODEL_PATH}")
            # For GGUF models, use llama-cpp-python
            from llama_cpp import Llama
            
            # Load the model with appropriate parameters
            model = Llama(
                model_path=MISTRAL_MODEL_PATH,
                n_ctx=4096,  # Context window size
                n_threads=4   # Number of CPU threads to use
            )
            return model
        else:
            logging.error(f"Mistral model not found at {MISTRAL_MODEL_PATH}")
            return None
    except Exception as e:
        logging.error(f"Error loading Mistral model: {str(e)}")
        return None

def generate_answer(query, context, model):
    """Generate an answer using the Mistral model"""
    try:
        # Create the prompt
        prompt = f"""<s>[INST] You are a legal assistant specialized in Sri Lankan building regulations and urban development.
Answer the following question based ONLY on the provided context. 
If the context doesn't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer: [/INST]"""
        
        # Using llama-cpp
        result = model.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.15,
            stop=["</s>", "[INST]"],
            echo=False
        )
        
        # Extract the generated text
        answer = result['choices'][0]['text'].strip()
        
        return answer, result.get('usage', {}).get('total_tokens', 0)
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}", 0

def rag_pipeline(query, embedding_model_path=EMBEDDING_MODEL_PATH, k=5):
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents
    2. Generate an answer using the Mistral model
    
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
        "tokens": tokens
    }

def interactive_mode():
    """
    Interactive CLI for the RAG system.
    """
    print("\n=== Sri Lankan Urban Development Regulations Assistant (Mistral with Pre-trained Embeddings) ===")
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
        output_file = os.path.join(RESULTS_DIR, 'batch_results.json')
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
