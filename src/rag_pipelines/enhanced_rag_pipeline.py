import os
import json
import logging
import time
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Add the directory containing the embedding pipelines to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pre_trained_embedding_models_with_LLMs'))

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")
ENHANCED_RESULTS_DIR = os.path.join(RESULTS_DIR, "enhanced_rag")
os.makedirs(ENHANCED_RESULTS_DIR, exist_ok=True)

# Model paths and configurations
MISTRAL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MRL_EMBEDDING_MODEL = "/Users/vishmiherath/Documents/FYP/data/my-mrl-embeddings"
MRL_DIMENSION = 256  # Default MRL dimension to use

# Groq API configuration
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"

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

def generate_reasoning_with_deepseek(query, retrieved_docs, max_tokens=1024, temperature=0.1):
    """
    Generate reasoning using the Groq API with DeepSeek-R1 model based on retrieved documents.
    """
    start_time = time.time()
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    
    # Create the prompt with specific instructions for reasoning
    system_message = """You are a legal reasoning assistant specialized in Sri Lankan building regulations and urban development.
Your task is to analyze the provided context and generate a detailed reasoning about the question.
Focus on:
1. Identifying the key legal principles and regulations relevant to the question
2. Analyzing how these principles apply to the specific scenario
3. Providing a structured reasoning process
4. Highlighting any ambiguities or areas where more information might be needed

Provide ONLY your reasoning based on the context. Do not make up information not present in the context.
If the context doesn't contain relevant information, explain what information would be needed to answer properly."""
    
    # Prepare the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide your detailed legal reasoning:"}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        # Make the API request
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract the reasoning from the response
        result = response.json()
        reasoning = result["choices"][0]["message"]["content"]
        
        # Calculate generation time and token usage
        generation_time = time.time() - start_time
        tokens_used = result.get("usage", {}).get("total_tokens", 0)
        
        return reasoning, {
            "time": generation_time,
            "tokens": tokens_used
        }
    except Exception as e:
        logging.error(f"Error generating reasoning with DeepSeek-R1: {str(e)}")
        return f"Error generating reasoning: {str(e)}", {
            "time": time.time() - start_time,
            "tokens": 0
        }

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

def generate_answer_with_mistral(query, context, reasoning, model):
    """Generate an answer using the Mistral model with enhanced context"""
    try:
        # Create the prompt with both original context and DeepSeek reasoning
        prompt = f"""<s>[INST] You are a legal assistant specialized in Sri Lankan building regulations and urban development.
Answer the following question based ONLY on the provided context and reasoning. 
If the context and reasoning don't contain relevant information, say "I don't have enough information to answer this question."

Context:
{context}

Expert Legal Reasoning:
{reasoning}

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
        logging.error(f"Error generating answer with Mistral: {str(e)}")
        return f"Error generating answer: {str(e)}", 0

def enhanced_rag_pipeline(query, k=5, dimension=None):
    """
    Enhanced RAG pipeline with reasoning:
    1. Retrieve relevant documents using MRL embeddings
    2. Generate reasoning with DeepSeek-R1
    3. Generate final answer with Mistral using both context and reasoning
    """
    # Use specified dimension or default
    dim = dimension if dimension is not None else MRL_DIMENSION
    
    # Step 1: Load vector database
    db = load_vector_db(dimension=dim)
    if not db:
        return "Error: Could not load vector database.", None, None, None
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents using MRL embeddings ({dim}d)")
    retrieved_docs = search_documents(query, MRL_EMBEDDING_MODEL, db, dimension=dim, k=k)
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    
    # Step 3: Generate reasoning with DeepSeek-R1
    logging.info(f"Generating reasoning with DeepSeek-R1 ({DEEPSEEK_MODEL})")
    reasoning, reasoning_info = generate_reasoning_with_deepseek(query, retrieved_docs)
    
    # Step 4: Load Mistral model
    mistral_model = load_mistral_model()
    if not mistral_model:
        return "Error: Could not load Mistral model.", context, reasoning, None
    
    # Step 5: Generate final answer with Mistral
    logging.info("Generating final answer with Mistral using enhanced context")
    answer, tokens = generate_answer_with_mistral(query, context, reasoning, mistral_model)
    
    return answer, context, reasoning, {
        "reasoning_time": reasoning_info.get("time", 0),
        "reasoning_tokens": reasoning_info.get("tokens", 0),
        "answer_tokens": tokens
    }

def load_test_queries():
    """Load test queries from test.json"""
    test_file = "test.json"
    
    if not os.path.exists(test_file):
        logging.error(f"Test file not found at {test_file}")
        return []
    
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Extract just the questions
        queries = [item['question'] for item in test_data]
        
        # Limit to a smaller set for testing
        return queries[:10]  # Adjust as needed
    except Exception as e:
        logging.error(f"Error loading test queries: {str(e)}")
        return []

def run_enhanced_rag_evaluation():
    """Run evaluation of the enhanced RAG pipeline on test queries"""
    # Load test queries
    test_queries = load_test_queries()
    if not test_queries:
        logging.error("No test queries found.")
        return
    
    logging.info(f"Loaded {len(test_queries)} test queries")
    
    # Prepare for results
    results = {}
    
    # Process each query
    for query in tqdm(test_queries, desc="Processing queries"):
        logging.info(f"Processing query: {query}")
        
        # Run enhanced RAG pipeline
        answer, context, reasoning, info = enhanced_rag_pipeline(query)
        
        # Store results
        results[query] = {
            "answer": answer,
            "context": context,
            "reasoning": reasoning,
            "info": info
        }
    
    # Save results to JSON
    results_file = os.path.join(ENHANCED_RESULTS_DIR, "enhanced_rag_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    generate_markdown_report(results)
    
    logging.info(f"Results saved to {results_file}")
    return results

def generate_markdown_report(results):
    """Generate a markdown report for the enhanced RAG results"""
    report_file = os.path.join(ENHANCED_RESULTS_DIR, "enhanced_rag_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Enhanced RAG Pipeline Evaluation Report\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report evaluates an enhanced RAG pipeline with the following components:\n\n")
        f.write("1. **Retrieval**: MRL embeddings (256d) for document retrieval\n")
        f.write(f"2. **Reasoning**: DeepSeek-R1 ({DEEPSEEK_MODEL}) for legal reasoning\n")
        f.write("3. **Answer Generation**: Mistral-7B for final answer generation\n\n")
        
        # Performance summary
        f.write("## Performance Summary\n\n")
        
        total_reasoning_tokens = sum(
            results[q]['info'].get('reasoning_tokens', 0) 
            for q in results if results[q].get('info')
        )
        total_answer_tokens = sum(
            results[q]['info'].get('answer_tokens', 0) 
            for q in results if results[q].get('info')
        )
        
        avg_reasoning_time = np.mean([
            results[q]['info'].get('reasoning_time', 0) 
            for q in results if results[q].get('info')
        ])
        
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Reasoning Tokens | {total_reasoning_tokens} |\n")
        f.write(f"| Total Answer Tokens | {total_answer_tokens} |\n")
        f.write(f"| Avg. Reasoning Time | {avg_reasoning_time:.2f}s |\n\n")
        
        # List the test queries used
        f.write("## Test Queries\n\n")
        for i, query in enumerate(results.keys()):
            f.write(f"{i+1}. {query}\n")
        f.write("\n")
        
        # Write detailed results for each query
        for query in results.keys():
            f.write(f"## Query: \"{query}\"\n\n")
            
            # DeepSeek reasoning
            f.write("### Legal Reasoning (DeepSeek-R1)\n\n")
            f.write(f"```\n{results[query]['reasoning']}\n```\n\n")
            
            if results[query].get('info'):
                tokens = results[query]['info'].get('reasoning_tokens', 0)
                time_taken = results[query]['info'].get('reasoning_time', 0)
                f.write(f"*Tokens: {tokens}, Time: {time_taken:.2f}s*\n\n")
            
            # Mistral answer
            f.write("### Final Answer (Mistral)\n\n")
            f.write(f"```\n{results[query]['answer']}\n```\n\n")
            
            if results[query].get('info'):
                tokens = results[query]['info'].get('answer_tokens', 0)
                f.write(f"*Tokens: {tokens}*\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Report generated at {report_file}")

def interactive_mode():
    """Interactive mode for testing the enhanced RAG pipeline"""
    global MRL_DIMENSION
    
    print("\n=== Enhanced RAG Pipeline with DeepSeek-R1 Reasoning ===")
    print(f"Using MRL embeddings ({MRL_DIMENSION}d) for retrieval")
    print(f"Using {DEEPSEEK_MODEL} for legal reasoning")
    print(f"Using Mistral-7B for final answer generation")
    print("\nType 'exit' to quit")
    print("Type 'dimension X' to change MRL dimension (e.g., 'dimension 128')")
    print("Type 'save' to save the last result to a file")
    
    last_result = None
    current_dimension = MRL_DIMENSION
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        
        # Handle dimension change command
        if query.lower().startswith('dimension '):
            try:
                new_dim = int(query.split()[1])
                if new_dim in [64, 128, 256, 512, 768]:
                    current_dimension = new_dim
                    MRL_DIMENSION = new_dim
                    print(f"\nChanged MRL dimension to {current_dimension}d")
                else:
                    print("\nSupported dimensions: 64, 128, 256, 512, 768")
                continue
            except (IndexError, ValueError):
                print("\nInvalid dimension format. Use 'dimension X' where X is a number.")
                continue
        
        # Handle save command
        if query.lower() == 'save' and last_result:
            save_path = os.path.join(ENHANCED_RESULTS_DIR, f"result_{int(time.time())}.json")
            with open(save_path, 'w') as f:
                json.dump(last_result, f, indent=2)
            print(f"\nSaved last result to {save_path}")
            continue
        
        print(f"\nProcessing your query using MRL {current_dimension}d embeddings...")
        start_time = time.time()
        answer, context, reasoning, info = enhanced_rag_pipeline(query, dimension=current_dimension)
        total_time = time.time() - start_time
        
        # Store the result
        last_result = {
            "query": query,
            "answer": answer,
            "reasoning": reasoning,
            "context": context,
            "info": info,
            "total_time": total_time,
            "dimension": current_dimension
        }
        
        # Display the reasoning
        print("\n" + "="*80)
        print("LEGAL REASONING (DeepSeek-R1)")
        print("="*80)
        print(reasoning)
        
        if info:
            print(f"\nReasoning tokens: {info.get('reasoning_tokens', 0)}, Time: {info.get('reasoning_time', 0):.2f}s")
        
        # Display the final answer
        print("\n" + "="*80)
        print("FINAL ANSWER (Mistral)")
        print("="*80)
        print(answer)
        
        # Display performance metrics
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        if info:
            print(f"Retrieval dimension: {current_dimension}d")
            print(f"Reasoning tokens: {info.get('reasoning_tokens', 0)}")
            print(f"Answer tokens: {info.get('answer_tokens', 0)}")
            print(f"Reasoning time: {info.get('reasoning_time', 0):.2f}s")
            print(f"Total processing time: {total_time:.2f}s")

if __name__ == "__main__":
    # Default to interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Run batch evaluation
        run_enhanced_rag_evaluation()
    else:
        # Run in interactive mode
        interactive_mode()
