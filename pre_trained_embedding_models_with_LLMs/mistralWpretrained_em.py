import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging
from llama_cpp import Llama
import time

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

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
        index = faiss.read_index(f"{index_path}.index")
        
        with open(f"{index_path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        return {
            'index': index,
            'documents': data['documents'],
            'sources': data['sources'],
            'types': data.get('types', ['unknown'] * len(data['documents']))
        }
    except Exception as e:
        logging.error(f"Error loading vector database: {str(e)}")
        return None

def search_documents(query, model_path, db, k=5):
    """
    Search the vector database for relevant documents.
    """
    model = SentenceTransformer(model_path)
    
    # Create query embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search
    distances, indices = db['index'].search(query_embedding.astype(np.float32), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        # Convert distance to similarity score (0-1 range)
        similarity = 1 - min(distances[0][i], 2) / 2  # Cap at 2 to keep in 0-1 range
        
        results.append({
            'similarity': similarity,
            'content': db['documents'][idx],
            'source': db['sources'][idx],
            'type': db['types'][idx]
        })
    
    return results

def load_llm(model_path):
    """
    Load the Mistral quantized model.
    """
    logging.info(f"Loading LLM from {model_path}")
    try:
        # n_ctx controls the context window size
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,        # Context window
            n_batch=512,       # Batch size for prompt processing
            n_gpu_layers=0     # Use CPU only, set to higher number if you have a GPU
        )
        logging.info("LLM loaded successfully")
        return llm
    except Exception as e:
        logging.error(f"Error loading LLM: {str(e)}")
        return None

def generate_answer(llm, query, retrieved_docs, max_tokens=1024, temperature=0.1):
    """
    Generate an answer using the LLM based on retrieved documents.
    """
    # Create context from retrieved documents
    context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(retrieved_docs)])
    
    # Create prompt
    prompt = f"""<s>[INST] You are an assistant specializing in Sri Lankan urban development regulations.

Answer the following question based on the context provided.
Answer the question thoroughly and list any applicable rules, clauses, or sections.

IMPORTANT INSTRUCTIONS:
1. If the exact answer isn't in the context but you can reasonably infer it from the provided regulations, provide that inference.
2. If you mention regulation numbers or refer to specific rules, cite them directly from the context.
3. If the information is partially present, provide what you can find and note what's missing.
4. Only say "I don't have information about that in the regulations" if the context has absolutely no relevant information.
5. Keep your answer concise and focused on the question.

Context information:
{context}

User question: {query}

Please provide a concise, accurate answer based solely on the provided context. If the context doesn't contain enough information to answer the question fully, acknowledge the limitations. [/INST]
"""

    # Generate answer
    start_time = time.time()
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        echo=False
    )
    end_time = time.time()
    
    answer = response["choices"][0]["text"].strip()
    
    generation_info = {
        "tokens": response["usage"]["completion_tokens"],
        "time": end_time - start_time
    }
    
    return answer, generation_info

def rag_pipeline(query, embedding_model_path, llm_model_path, index_path="legal_faiss_index", k=5):
    """
    Complete RAG pipeline:
    1. Retrieve relevant documents using your custom embedding model
    2. Generate an answer using the Mistral model
    """
    # Step 1: Load vector database or create if it doesn't exist
    if not os.path.exists(f"{index_path}.index"):
        logging.info("Vector database not found. Creating...")
        db = create_vector_db(
            text_folder="./data/cleaned_txt",
            tables_folder="./tables",
            model_path=embedding_model_path,
            index_path=index_path
        )
    else:
        logging.info("Loading existing vector database...")
        db = load_vector_db(index_path)
    
    if not db:
        return "Error: Could not load or create vector database.", None, None
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents for query: {query}")
    search_results = search_documents(query, embedding_model_path, db, k=k)
    
    # Step 3: Load LLM
    llm = load_llm(llm_model_path)
    if not llm:
        return "Error: Could not load language model.", search_results, None
    
    # Step 4: Generate answer
    logging.info("Generating answer...")
    answer, generation_info = generate_answer(llm, query, search_results)
    
    logging.info(f"Answer generated with {generation_info['tokens']} tokens in {generation_info['time']:.2f} seconds")
    
    return answer, search_results, generation_info

# Compare custom embeddings vs. baseline
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

# Interactive CLI
def interactive_mode(embedding_model_path, llm_model_path, index_path="legal_faiss_index"):
    """
    Interactive CLI for the RAG system.
    """
    print("\n=== Sri Lankan Urban Development Regulations Assistant ===")
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
            llm_model_path=llm_model_path,
            index_path=index_path
        )
        
        print("\n=== Retrieved Documents ===")
        for i, doc in enumerate(results):
            print(f"{i+1}. [Score: {doc['similarity']:.4f}] Source: {os.path.basename(doc['source'])}")
            print(f"   Excerpt: {doc['content'][:150]}...\n")
        
        print("\n=== Generated Answer ===")
        print(answer)
        
        if info:
            print(f"\nGeneration time: {info['time']:.2f} seconds, Tokens used: {info['tokens']}")

if __name__ == "__main__":
    # Configuration
    EMBEDDING_MODEL_PATH = "legal-embeddings-model"  # Your fine-tuned embedding model
    LLM_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Path to Mistral model
    
    # Create vector database if it doesn't exist
    if not os.path.exists("legal_faiss_index.index"):
        create_vector_db(
            text_folder="data/cleaned_txt",  # Update with your folder path
            tables_folder="data/table_data",          # Update with your folder path
            model_path=EMBEDDING_MODEL_PATH
        )
    
    # Start interactive mode
    interactive_mode(EMBEDDING_MODEL_PATH, LLM_MODEL_PATH)