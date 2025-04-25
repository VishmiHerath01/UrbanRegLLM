import os
import time
import json
import numpy as np
import pandas as pd
import faiss
import pickle
import logging
import psutil
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
CLEANED_TEXT_DIR = os.path.join(DATA_DIR, "cleaned_txt")
TABLES_DIR = os.path.join(DATA_DIR, "table_data")

# Full absolute paths to the models
REGULAR_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/legal-embeddings-model"
MRL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/data/my-mrl-embeddings"

# Create results directory
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load test queries from test.json
def load_test_queries():
    """Load test queries from test.json file"""
    try:
        test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "test.json")
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [item['question'] for item in data]
    except Exception as e:
        logging.error(f"Error loading test queries: {str(e)}")
        return []

def create_vector_db(text_folder, tables_folder, model_path, index_path, dimension=None):
    """
    Create a vector database using the specified embedding model.
    
    Args:
        text_folder: Folder containing text documents
        tables_folder: Folder containing table documents
        model_path: Path to the embedding model
        index_path: Path to save the FAISS index
        dimension: For MRL embeddings, the dimension to truncate to
    """
    start_time = time.time()
    
    # Load the model
    logging.info(f"Loading embedding model from {model_path}...")
    model = SentenceTransformer(model_path)
    
    # Get the embedding dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    logging.info(f"Model embedding dimension: {embedding_dim}")
    
    # If dimension is specified and less than the model's dimension, we'll truncate
    # This is for MRL embeddings
    if dimension and dimension < embedding_dim:
        logging.info(f"Will truncate embeddings to {dimension} dimensions")
        embedding_dim = dimension
    
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
    
    # Track memory usage before embedding
    memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # Generate embeddings
    embeddings = model.encode(documents, convert_to_numpy=True)
    
    # Track memory usage after embedding
    memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    
    # Truncate embeddings if needed (for MRL)
    if dimension and dimension < embeddings.shape[1]:
        logging.info(f"Truncating embeddings from {embeddings.shape[1]}d to {dimension}d")
        embeddings = embeddings[:, :dimension]
    
    # Create FAISS index
    logging.info(f"Creating FAISS index with dimension {embedding_dim}...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings.astype(np.float32))
    
    # Save the index and metadata
    if dimension:
        index_path = f"{index_path}_{dimension}d"
    
    logging.info(f"Saving index to {index_path}...")
    faiss.write_index(index, f"{index_path}.index")
    
    # Save document data
    with open(f"{index_path}.pkl", 'wb') as f:
        pickle.dump({
            'documents': documents,
            'sources': sources,
            'types': types,
            'dimension': embedding_dim
        }, f)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Vector database created in {elapsed_time:.2f} seconds")
    logging.info(f"Memory used for embedding: {memory_used:.2f} MB")
    
    return {
        'index': index,
        'documents': documents,
        'sources': sources,
        'types': types,
        'embedding_time': elapsed_time,
        'memory_used': memory_used,
        'dimension': embedding_dim
    }

def load_vector_db(model_type, dimension=None):
    """
    Load a vector database from disk.
    
    Args:
        model_type: 'regular' or 'mrl'
        dimension: For MRL, the dimension to load
    """
    if model_type == 'regular':
        index_path = "regular_legal_faiss_index"
    else:  # mrl
        index_path = "mrl_legal_faiss_index"
        if dimension:
            index_path = f"{index_path}_{dimension}d"
    
    try:
        logging.info(f"Loading index from {index_path}...")
        index = faiss.read_index(f"{index_path}.index")
        
        with open(f"{index_path}.pkl", 'rb') as f:
            data = pickle.load(f)
        
        return {
            'index': index,
            'documents': data['documents'],
            'sources': data['sources'],
            'types': data['types'],
            'dimension': data.get('dimension')
        }
    except Exception as e:
        logging.error(f"Error loading vector database: {str(e)}")
        return None

def search_documents(query, model_path, db, dimension=None, k=5):
    """
    Search for documents using the specified model.
    
    Args:
        query: The search query
        model_path: Path to the embedding model
        db: The vector database
        dimension: For MRL, the dimension to use
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

def compare_embeddings():
    """
    Compare regular and MRL embeddings on various metrics.
    """
    # Ensure the results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load test queries
    test_queries = load_test_queries()
    if not test_queries:
        logging.error("No test queries found.")
        return
    
    logging.info(f"Loaded {len(test_queries)} test queries")
    
    # Create or load vector databases
    regular_db = load_vector_db('regular')
    if not regular_db:
        logging.info("Creating regular vector database...")
        regular_db = create_vector_db(
            CLEANED_TEXT_DIR,
            TABLES_DIR,
            REGULAR_MODEL_PATH,
            "regular_legal_faiss_index"
        )
    
    # MRL dimensions to test
    mrl_dimensions = [64, 128, 256, 512, 768]
    mrl_dbs = {}
    
    for dim in mrl_dimensions:
        mrl_db = load_vector_db('mrl', dimension=dim)
        if not mrl_db:
            logging.info(f"Creating MRL vector database with dimension {dim}...")
            mrl_db = create_vector_db(
                CLEANED_TEXT_DIR,
                TABLES_DIR,
                MRL_MODEL_PATH,
                "mrl_legal_faiss_index",
                dimension=dim
            )
        mrl_dbs[dim] = mrl_db
    
    # Prepare results
    results = {
        'queries': [],
        'regular': {
            'search_times': [],
            'top_results': []
        }
    }
    
    for dim in mrl_dimensions:
        results[f'mrl_{dim}d'] = {
            'search_times': [],
            'top_results': []
        }
    
    # Run searches for each query
    for query in test_queries:
        logging.info(f"Searching for: {query}")
        
        results['queries'].append(query)
        
        # Search with regular embeddings
        regular_results, regular_time = search_documents(
            query,
            REGULAR_MODEL_PATH,
            regular_db,
            k=5
        )
        
        results['regular']['search_times'].append(regular_time)
        results['regular']['top_results'].append(regular_results)
        
        # Search with MRL embeddings at different dimensions
        for dim in mrl_dimensions:
            mrl_results, mrl_time = search_documents(
                query,
                MRL_MODEL_PATH,
                mrl_dbs[dim],
                dimension=dim,
                k=5
            )
            
            results[f'mrl_{dim}d']['search_times'].append(mrl_time)
            results[f'mrl_{dim}d']['top_results'].append(mrl_results)
    
    # Save raw results
    with open(os.path.join(RESULTS_DIR, 'embedding_comparison_results.json'), 'w') as f:
        # Convert to a more JSON-friendly format
        json_results = {
            'queries': results['queries'],
            'regular': {
                'search_times': results['regular']['search_times'],
                # We can't directly serialize the top_results, so extract key info
                'top_results': [
                    [
                        {
                            'content_preview': r['content'][:100] + '...',
                            'source': r['source'],
                            'type': r['type'],
                            'similarity': r['similarity']
                        } for r in query_results
                    ] for query_results in results['regular']['top_results']
                ]
            }
        }
        
        for dim in mrl_dimensions:
            json_results[f'mrl_{dim}d'] = {
                'search_times': results[f'mrl_{dim}d']['search_times'],
                'top_results': [
                    [
                        {
                            'content_preview': r['content'][:100] + '...',
                            'source': r['source'],
                            'type': r['type'],
                            'similarity': r['similarity']
                        } for r in query_results
                    ] for query_results in results[f'mrl_{dim}d']['top_results']
                ]
            }
        
        json.dump(json_results, f, indent=2)
    
    # Generate comparison report
    generate_comparison_report(results, mrl_dimensions)
    
    # Generate visualizations
    generate_visualizations(results, mrl_dimensions)
    
    return results

def generate_comparison_report(results, mrl_dimensions):
    """
    Generate a markdown report comparing the embedding approaches.
    """
    report_path = os.path.join(RESULTS_DIR, 'comparison_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Embedding Comparison Report\n\n")
        
        # Overview
        f.write("## Overview\n\n")
        f.write("This report compares regular legal embeddings with Matryoshka Representation Learning (MRL) embeddings ")
        f.write("at various dimensions for legal document retrieval.\n\n")
        
        # Performance metrics
        f.write("## Performance Metrics\n\n")
        
        # Search speed
        f.write("### Search Speed\n\n")
        f.write("Average search time in seconds:\n\n")
        f.write("| Model | Average Search Time (s) |\n")
        f.write("|-------|-------------------------|\n")
        
        avg_regular_time = np.mean(results['regular']['search_times'])
        f.write(f"| Regular | {avg_regular_time:.6f} |\n")
        
        for dim in mrl_dimensions:
            avg_mrl_time = np.mean(results[f'mrl_{dim}d']['search_times'])
            f.write(f"| MRL ({dim}d) | {avg_mrl_time:.6f} |\n")
        
        f.write("\n")
        
        # Memory usage
        f.write("### Memory Usage\n\n")
        f.write("| Model | Dimension | Index Size |\n")
        f.write("|-------|-----------|------------|\n")
        
        # Get index file sizes
        regular_index_size = os.path.getsize("regular_legal_faiss_index.index") / (1024 * 1024)  # MB
        f.write(f"| Regular | 768 | {regular_index_size:.2f} MB |\n")
        
        for dim in mrl_dimensions:
            mrl_index_path = f"mrl_legal_faiss_index_{dim}d.index"
            if os.path.exists(mrl_index_path):
                mrl_index_size = os.path.getsize(mrl_index_path) / (1024 * 1024)  # MB
                f.write(f"| MRL | {dim} | {mrl_index_size:.2f} MB |\n")
        
        f.write("\n")
        
        # Sample retrieval results
        f.write("## Sample Retrieval Results\n\n")
        
        # Show results for a few sample queries
        sample_indices = np.random.choice(len(results['queries']), min(5, len(results['queries'])), replace=False)
        
        for idx in sample_indices:
            query = results['queries'][idx]
            f.write(f"### Query: \"{query}\"\n\n")
            
            # Regular results
            f.write("#### Regular Embeddings\n\n")
            regular_results = results['regular']['top_results'][idx]
            for i, result in enumerate(regular_results[:3]):  # Show top 3
                f.write(f"**Result {i+1}** (Similarity: {result['similarity']:.4f})\n\n")
                f.write(f"Source: {result['source']} ({result['type']})\n\n")
                f.write(f"```\n{result['content'][:300]}...\n```\n\n")
            
            # MRL results for each dimension
            for dim in mrl_dimensions:
                f.write(f"#### MRL Embeddings ({dim}d)\n\n")
                mrl_results = results[f'mrl_{dim}d']['top_results'][idx]
                for i, result in enumerate(mrl_results[:3]):  # Show top 3
                    f.write(f"**Result {i+1}** (Similarity: {result['similarity']:.4f})\n\n")
                    f.write(f"Source: {result['source']} ({result['type']})\n\n")
                    f.write(f"```\n{result['content'][:300]}...\n```\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Comparison report generated at {report_path}")

def generate_visualizations(results, mrl_dimensions):
    """
    Generate visualizations comparing the embedding approaches.
    """
    # Search time comparison
    plt.figure(figsize=(10, 6))
    
    # Regular embeddings
    avg_regular_time = np.mean(results['regular']['search_times'])
    plt.bar('Regular', avg_regular_time, color='blue', alpha=0.7)
    
    # MRL embeddings at different dimensions
    for dim in mrl_dimensions:
        avg_mrl_time = np.mean(results[f'mrl_{dim}d']['search_times'])
        plt.bar(f'MRL ({dim}d)', avg_mrl_time, color='green', alpha=0.7)
    
    plt.title('Average Search Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, 'search_time_comparison.png'))
    plt.close()
    
    # Memory usage comparison
    plt.figure(figsize=(10, 6))
    
    # Get index file sizes
    regular_index_size = os.path.getsize("regular_legal_faiss_index.index") / (1024 * 1024)  # MB
    plt.bar('Regular', regular_index_size, color='blue', alpha=0.7)
    
    for dim in mrl_dimensions:
        mrl_index_path = f"mrl_legal_faiss_index_{dim}d.index"
        if os.path.exists(mrl_index_path):
            mrl_index_size = os.path.getsize(mrl_index_path) / (1024 * 1024)  # MB
            plt.bar(f'MRL ({dim}d)', mrl_index_size, color='green', alpha=0.7)
    
    plt.title('Index Size Comparison')
    plt.ylabel('Size (MB)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, 'memory_usage_comparison.png'))
    plt.close()
    
    logging.info(f"Visualizations saved to {RESULTS_DIR}")

def interactive_mode():
    """
    Interactive mode for testing and comparing embeddings.
    """
    print("\n=== Legal Document Embedding Comparison ===")
    print("Type 'exit' to quit, 'compare' to run full comparison\n")
    
    # Load or create vector databases
    regular_db = load_vector_db('regular')
    if not regular_db:
        print("Creating regular vector database...")
        regular_db = create_vector_db(
            CLEANED_TEXT_DIR,
            TABLES_DIR,
            REGULAR_MODEL_PATH,
            "regular_legal_faiss_index"
        )
    
    # Load MRL database with default dimension
    mrl_dimension = 256  # Default dimension
    mrl_db = load_vector_db('mrl', dimension=mrl_dimension)
    if not mrl_db:
        print(f"Creating MRL vector database with dimension {mrl_dimension}...")
        mrl_db = create_vector_db(
            CLEANED_TEXT_DIR,
            TABLES_DIR,
            MRL_MODEL_PATH,
            "mrl_legal_faiss_index",
            dimension=mrl_dimension
        )
    
    while True:
        query = input("\nEnter your query (or 'exit'/'compare'): ")
        
        if query.lower() == 'exit':
            break
        elif query.lower() == 'compare':
            print("\nRunning full comparison...")
            compare_embeddings()
            print("Comparison complete. Results saved to the comparison_results directory.")
            continue
        elif query.lower().startswith('dimension '):
            try:
                new_dim = int(query.split()[1])
                if new_dim in [64, 128, 256, 512, 768]:
                    mrl_dimension = new_dim
                    mrl_db = load_vector_db('mrl', dimension=mrl_dimension)
                    if not mrl_db:
                        print(f"Creating MRL vector database with dimension {mrl_dimension}...")
                        mrl_db = create_vector_db(
                            CLEANED_TEXT_DIR,
                            TABLES_DIR,
                            MRL_MODEL_PATH,
                            "mrl_legal_faiss_index",
                            dimension=mrl_dimension
                        )
                    print(f"Changed MRL dimension to {mrl_dimension}")
                else:
                    print("Supported dimensions: 64, 128, 256, 512, 768")
                continue
            except:
                print("Invalid dimension format. Use 'dimension X' where X is a number.")
                continue
        
        print("\nSearching with regular embeddings...")
        regular_results, regular_time = search_documents(
            query,
            REGULAR_MODEL_PATH,
            regular_db,
            k=5
        )
        
        print(f"Search completed in {regular_time:.6f} seconds\n")
        print("Top results:")
        for i, result in enumerate(regular_results):
            print(f"{i+1}. {result['source']} - Similarity: {result['similarity']:.4f}")
            print(f"   {result['content'][:100]}...\n")
        
        print(f"\nSearching with MRL embeddings ({mrl_dimension}d)...")
        mrl_results, mrl_time = search_documents(
            query,
            MRL_MODEL_PATH,
            mrl_db,
            dimension=mrl_dimension,
            k=5
        )
        
        print(f"Search completed in {mrl_time:.6f} seconds\n")
        print("Top results:")
        for i, result in enumerate(mrl_results):
            print(f"{i+1}. {result['source']} - Similarity: {result['similarity']:.4f}")
            print(f"   {result['content'][:100]}...\n")
        
        print("\nPerformance comparison:")
        print(f"Regular: {regular_time:.6f}s vs MRL ({mrl_dimension}d): {mrl_time:.6f}s")
        if mrl_time < regular_time:
            print(f"MRL is {(regular_time/mrl_time - 1)*100:.2f}% faster")
        else:
            print(f"Regular is {(mrl_time/regular_time - 1)*100:.2f}% faster")

if __name__ == "__main__":
    # Check if we should run in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Run the comparison
        compare_embeddings()
