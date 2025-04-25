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


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")
ENHANCED_RESULTS_DIR = os.path.join(RESULTS_DIR, "enhanced_rag_pretrained")
os.makedirs(ENHANCED_RESULTS_DIR, exist_ok=True)


MISTRAL_MODEL_PATH = "/Users/vishmiherath/Documents/FYP/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PRETRAINED_EMBEDDING_MODEL = "/Users/vishmiherath/Documents/FYP/legal-embeddings-model"  # Pre-trained model path


GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"

def load_vector_db(index_path="legal_faiss_index"):
    try:
        logging.info(f"Loading existing pre-trained vector database...")
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
        logging.error(f"Error loading pre-trained vector database: {str(e)}")
        return None

def search_documents(query, model_path, db, k=5):

    model = SentenceTransformer(model_path)
    

    query_embedding = model.encode([query], convert_to_numpy=True)
    

    distances, indices = db['index'].search(query_embedding.astype(np.float32), k=k)
    

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
    start_time = time.time()
    
    # Prepare context from retrieved documents
    context = ""
    for i, doc in enumerate(retrieved_docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    

    prompt = f"""You are a legal expert specializing in urban development regulations. 
I'll provide you with a question and relevant documents from the Urban Development Regulations.
Your task is to analyze these documents and provide detailed legal reasoning about how they apply to the question.
Focus on identifying specific requirements, constraints, and guidelines that are relevant.

Question: {query}

Relevant documents from Urban Development Regulations:
{context}

Please provide your detailed legal reasoning about how these regulations apply to the question:"""

    # Prepare the API request
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are a legal expert specializing in urban development regulations."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        # Make the API request
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        # Extract the reasoning from the response
        result = response.json()
        reasoning = result["choices"][0]["message"]["content"]
        
        # Calculate tokens and time
        tokens = result["usage"]["completion_tokens"]
        reasoning_time = time.time() - start_time
        
        return reasoning, {"tokens": tokens, "time": reasoning_time}
    except Exception as e:
        logging.error(f"Error generating reasoning: {str(e)}")
        return f"Error generating legal reasoning: {str(e)}", {"tokens": 0, "time": time.time() - start_time}

def load_mistral_model():
    try:
        logging.info(f"Loading Mistral model from {MISTRAL_MODEL_PATH}")
        model = Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_ctx=4096,  
            n_threads=4,  
            n_gpu_layers=0  
        )
        return model
    except Exception as e:
        logging.error(f"Error loading Mistral model: {str(e)}")
        return None

def generate_answer_with_mistral(query, context, reasoning, model):
    start_time = time.time()
    
    # Create the prompt with both context and reasoning
    prompt = f"""<s>[INST] You are an expert assistant specializing in urban development regulations. 
Answer the following question about urban development regulations based on the provided context and legal reasoning.
Be concise, accurate, and directly address the question.

Question: {query}

Context from Urban Development Regulations:
{context}

Legal Reasoning:
{reasoning}

Provide a clear, concise answer to the question based on the context and legal reasoning: [/INST]"""

    try:
        # Generate response
        response = model.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repeat_penalty=1.1,
            top_k=40,
            stop=["</s>", "[INST]"]
        )
        
        # Extract answer and count tokens
        answer = response["choices"][0]["text"].strip()
        tokens = len(answer.split())
        
        return answer, tokens
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return f"Error generating answer: {str(e)}", 0

def enhanced_rag_pipeline(query, k=5):
    # Step 1: Load vector database
    db = load_vector_db()
    if not db:
        return "Error: Could not load vector database.", None, None, None
    
    # Step 2: Search for relevant documents
    logging.info(f"Searching for relevant documents using pre-trained embeddings")
    retrieved_docs = search_documents(query, PRETRAINED_EMBEDDING_MODEL, db, k=k)
    
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
    try:
        test_path = os.path.join(DATA_DIR, "datasets", "splits", "test.json")
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Extract questions
        queries = [item['question'] for item in test_data if 'question' in item]
        
        # Limit to first 10 for faster evaluation
        return queries[:10]
    except Exception as e:
        logging.error(f"Error loading test queries: {str(e)}")
        return [
            "What are the requirements for natural light and ventilation in buildings?",
            "What are the minimum plot coverage requirements for buildings?",
            "How are blind walls and boundary structures regulated for safety?"
        ]

def run_enhanced_rag_evaluation():
    # Load test queries
    queries = load_test_queries()
    logging.info(f"Running evaluation on {len(queries)} test queries")
    
    # Results container
    results = {}
    
    # Process each query
    for query in tqdm(queries, desc="Processing queries"):
        start_time = time.time()
        
        # Run the pipeline
        answer, context, reasoning, info = enhanced_rag_pipeline(query)
        
        # Record results
        total_time = time.time() - start_time
        results[query] = {
            "answer": answer,
            "reasoning": reasoning,
            "context": context,
            "info": info,
            "total_time": total_time
        }
        
        # Log progress
        logging.info(f"Processed query: {query[:50]}... in {total_time:.2f}s")
    
    # Save results
    results_file = os.path.join(ENHANCED_RESULTS_DIR, "pretrained_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    generate_markdown_report(results)
    
    return results

def generate_markdown_report(results):
    report_file = os.path.join(ENHANCED_RESULTS_DIR, "pretrained_report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Enhanced RAG Pipeline with Pre-trained Embeddings Evaluation\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("- **Embedding Model**: Pre-trained Legal Embeddings\n")
        f.write(f"- **Reasoning Model**: {DEEPSEEK_MODEL}\n")
        f.write("- **Answer Generation Model**: Mistral-7B-Instruct (Quantized)\n\n")
        
        f.write("## Performance Summary\n\n")
        
        # Calculate average metrics
        total_time = 0
        reasoning_time = 0
        reasoning_tokens = 0
        answer_tokens = 0
        
        for query, data in results.items():
            total_time += data.get('total_time', 0)
            if data.get('info'):
                reasoning_time += data['info'].get('reasoning_time', 0)
                reasoning_tokens += data['info'].get('reasoning_tokens', 0)
                answer_tokens += data['info'].get('answer_tokens', 0)
        
        avg_time = total_time / len(results) if results else 0
        avg_reasoning_time = reasoning_time / len(results) if results else 0
        avg_reasoning_tokens = reasoning_tokens / len(results) if results else 0
        avg_answer_tokens = answer_tokens / len(results) if results else 0
        
        f.write(f"- **Average Total Time**: {avg_time:.2f}s\n")
        f.write(f"- **Average Reasoning Time**: {avg_reasoning_time:.2f}s\n")
        f.write(f"- **Average Reasoning Tokens**: {avg_reasoning_tokens:.1f}\n")
        f.write(f"- **Average Answer Tokens**: {avg_answer_tokens:.1f}\n\n")
        
        f.write("## Detailed Results\n\n")
        
        # Write detailed results for each query
        for query, data in results.items():
            f.write(f"### Query: \"{query}\"\n\n")
            
            # Performance metrics
            f.write("#### Performance Metrics\n\n")
            f.write(f"- Total Time: {data.get('total_time', 0):.2f}s\n")
            if data.get('info'):
                f.write(f"- Reasoning Time: {data['info'].get('reasoning_time', 0):.2f}s\n")
                f.write(f"- Reasoning Tokens: {data['info'].get('reasoning_tokens', 0)}\n")
                f.write(f"- Answer Tokens: {data['info'].get('answer_tokens', 0)}\n")
            f.write("\n")
            
            # Legal reasoning
            f.write("#### Legal Reasoning (DeepSeek-R1)\n\n")
            f.write(f"```\n{data.get('reasoning', 'No reasoning generated')}\n```\n\n")
            
            # Mistral answer
            f.write("### Final Answer (Mistral)\n\n")
            f.write(f"```\n{data.get('answer', 'No answer generated')}\n```\n\n")
            
            if data.get('info'):
                tokens = data['info'].get('answer_tokens', 0)
                f.write(f"*Tokens: {tokens}*\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Report generated at {report_file}")

def interactive_mode():
    print("\n=== Enhanced RAG Pipeline with DeepSeek-R1 Reasoning ===")
    print(f"Using pre-trained legal embeddings for retrieval")
    print(f"Using {DEEPSEEK_MODEL} for legal reasoning")
    print(f"Using Mistral-7B for final answer generation")
    print("\nType 'exit' to quit")
    print("Type 'save' to save the last result to a file")
    
    last_result = None
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() == 'exit':
            break
        

        if query.lower() == 'save' and last_result:
            save_path = os.path.join(ENHANCED_RESULTS_DIR, f"result_{int(time.time())}.json")
            with open(save_path, 'w') as f:
                json.dump(last_result, f, indent=2)
            print(f"\nSaved last result to {save_path}")
            continue
        
        print(f"\nProcessing your query using pre-trained legal embeddings...")
        start_time = time.time()
        answer, context, reasoning, info = enhanced_rag_pipeline(query)
        total_time = time.time() - start_time
        

        last_result = {
            "query": query,
            "answer": answer,
            "reasoning": reasoning,
            "context": context,
            "info": info,
            "total_time": total_time
        }
        

        print("\n" + "="*80)
        print("LEGAL REASONING (DeepSeek-R1)")
        print("="*80)
        print(reasoning)
        
        if info:
            print(f"\nReasoning tokens: {info.get('reasoning_tokens', 0)}, Time: {info.get('reasoning_time', 0):.2f}s")
        

        print("\n" + "="*80)
        print("FINAL ANSWER (Mistral)")
        print("="*80)
        print(answer)
        

        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        if info:
            print(f"Reasoning tokens: {info.get('reasoning_tokens', 0)}")
            print(f"Answer tokens: {info.get('answer_tokens', 0)}")
            print(f"Reasoning time: {info.get('reasoning_time', 0):.2f}s")
            print(f"Total processing time: {total_time:.2f}s")

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":

        run_enhanced_rag_evaluation()
    else:

        interactive_mode()
