import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import logging

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def create_vector_db(text_folder, tables_folder, model_path, index_path="legal_faiss_index"):

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

def search_vector_db(query, model_path, index_path="legal_faiss_index", k=3):
    
    #Search the vector database for similar documents.
    
    # Load model
    model = SentenceTransformer(model_path)
    
    # Load index
    index = faiss.read_index(f"{index_path}.index")
    
    with open(f"{index_path}.pkl", 'rb') as f:
        data = pickle.load(f)
    
    documents = data['documents']
    sources = data['sources']
    types = data.get('types', ['unknown'] * len(documents))
    
    # Create query embedding
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Search
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'score': distances[0][i],
            'content': documents[idx],
            'source': sources[idx],
            'type': types[idx]
        })
    
    return results

# Example usage
if __name__ == "__main__":
    #  Create the vector database
    run_create_db = True
    
    if run_create_db:
        db = create_vector_db(
            text_folder="data/cleaned_txt",
            tables_folder="data/tables",
            model_path="legal-embeddings-model"
        )
    
    # Test some queries
    run_test_queries = True
    
    if run_test_queries:
        test_queries = [
            "What are the requirements for changing the use of a building?",
            "What are the requirements for developments near expressways?",
            "How do the rainwater harvesting requirements vary across different rainfall zones in Sri Lanka?"
        ]
        
        for query in test_queries:
            results = search_vector_db(
                query, 
                model_path="legal-embeddings-model"
            )
            print(f"\nQuery: {query}")
            print("Top 3 retrieved chunks:")
            for i, result in enumerate(results):
                print(f"{i+1}. [Score: {result['score']:.4f}] {result['content'][:200]}... (Source: {result['source']})")