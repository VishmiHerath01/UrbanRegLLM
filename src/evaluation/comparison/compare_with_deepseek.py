import os
import json
import csv
import time
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import functions from your Groq DeepSeek-R1 pipelines
from rag_pipelines.pretrained.deepseekr1Wpretrained_em import rag_pipeline as regular_rag_pipeline, MODEL_NAME
from rag_pipelines.pretrained.deepseekr1Wmrl_em import rag_pipeline as mrl_rag_pipeline

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model paths
REGULAR_EMBEDDING_MODEL = "/Users/vishmiherath/Documents/FYP/legal-embeddings-model"
MRL_EMBEDDING_MODEL = "/Users/vishmiherath/Documents/FYP/data/my-mrl-embeddings"
MRL_DIMENSION = 256  # Default MRL dimension to use

def load_test_queries():
    """Load test queries from test.json"""
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "test.json")
    
    if not os.path.exists(test_file):
        logging.error(f"Test file not found at {test_file}")
        return []
    
    try:
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Extract just the questions
        queries = [item['question'] for item in test_data]
        
        # Limit to a smaller set for testing if needed
        # return queries[:10]
        return queries
    except Exception as e:
        logging.error(f"Error loading test queries: {str(e)}")
        return []

def get_context_from_results(docs):
    """Extract context from retrieved documents"""
    if not docs:
        return ""
    
    context = ""
    for i, doc in enumerate(docs):
        context += f"Document {i+1}:\n{doc['content']}\n\n"
    
    return context

def compare_with_deepseek():
    """Compare retrieval approaches by generating answers with Groq DeepSeek-R1"""
    # Load test queries
    test_queries = load_test_queries()
    if not test_queries:
        logging.error("No test queries found.")
        return
    
    logging.info(f"Loaded {len(test_queries)} test queries")
    
    # Prepare for results
    answers = {}
    
    # Generate answers for each query
    for query in tqdm(test_queries, desc="Processing queries"):
        logging.info(f"Generating answers for query: {query}")
        
        # Generate answer using regular embeddings with Groq DeepSeek-R1
        try:
            regular_answer, regular_docs, regular_info = regular_rag_pipeline(
                query,
                embedding_model_path=REGULAR_EMBEDDING_MODEL,
                k=5
            )
            regular_context = get_context_from_results(regular_docs)
            logging.info(f"Generated answer with regular embeddings: {len(regular_answer)} chars")
        except Exception as e:
            logging.error(f"Error with regular embeddings: {str(e)}")
            regular_answer = f"Error: {str(e)}"
            regular_context = ""
            regular_docs = []
            regular_info = None
        
        # Generate answer using MRL embeddings with Groq DeepSeek-R1
        try:
            mrl_answer, mrl_docs, mrl_info = mrl_rag_pipeline(
                query,
                model_path=MRL_EMBEDDING_MODEL,
                dimension=MRL_DIMENSION,
                k=5
            )
            mrl_context = get_context_from_results(mrl_docs)
            logging.info(f"Generated answer with MRL embeddings: {len(mrl_answer)} chars")
        except Exception as e:
            logging.error(f"Error with MRL embeddings: {str(e)}")
            mrl_answer = f"Error: {str(e)}"
            mrl_context = ""
            mrl_docs = []
            mrl_info = None
        
        # Store the results
        answers[query] = {
            "regular": {
                "answer": regular_answer,
                "context": regular_context,
                "metrics": regular_info
            },
            "mrl": {
                "answer": mrl_answer,
                "context": mrl_context,
                "metrics": mrl_info,
                "dimension": MRL_DIMENSION
            }
        }
    
    # Save the answers to a JSON file
    answers_file = os.path.join(RESULTS_DIR, "deepseek_answers.json")
    with open(answers_file, 'w') as f:
        json.dump(answers, f, indent=2)
    
    # Generate CSV for easier analysis
    csv_file = os.path.join(RESULTS_DIR, "deepseek_comparison.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Query", 
            "Regular Answer", "Regular Search Time", "Regular Generation Time", "Regular Tokens",
            f"MRL ({MRL_DIMENSION}d) Answer", f"MRL ({MRL_DIMENSION}d) Search Time", f"MRL ({MRL_DIMENSION}d) Generation Time", f"MRL ({MRL_DIMENSION}d) Tokens"
        ])
        
        for query, data in answers.items():
            writer.writerow([
                query,
                data["regular"]["answer"],
                data["regular"]["metrics"].get("search_time", "N/A") if data["regular"]["metrics"] else "N/A",
                data["regular"]["metrics"].get("generation_time", "N/A") if data["regular"]["metrics"] else "N/A",
                data["regular"]["metrics"].get("tokens", "N/A") if data["regular"]["metrics"] else "N/A",
                data["mrl"]["answer"],
                data["mrl"]["metrics"].get("search_time", "N/A") if data["mrl"]["metrics"] else "N/A",
                data["mrl"]["metrics"].get("generation_time", "N/A") if data["mrl"]["metrics"] else "N/A",
                data["mrl"]["metrics"].get("tokens", "N/A") if data["mrl"]["metrics"] else "N/A"
            ])
    
    # Generate a markdown report
    generate_markdown_report(answers)
    
    logging.info(f"Results saved to {answers_file} and {csv_file}")
    return answers

def generate_markdown_report(answers):
    """Generate a detailed markdown report comparing the answers"""
    report_path = os.path.join(RESULTS_DIR, "deepseek_answer_report.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Groq DeepSeek-R1 ({MODEL_NAME}) Answer Comparison Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"This report compares answers generated by Groq DeepSeek-R1 ({MODEL_NAME}) using two different embedding approaches:\n\n")
        f.write("1. **Regular Embeddings**: Using the full-dimensional legal embeddings model\n")
        f.write("2. **MRL Embeddings**: Using Matryoshka Representation Learning embeddings (256d)\n\n")
        
        # Performance metrics
        f.write("## Performance Metrics\n\n")
        
        # Calculate average metrics
        regular_search_times = [data["regular"]["metrics"].get("search_time", 0) for data in answers.values() if data["regular"]["metrics"]]
        regular_gen_times = [data["regular"]["metrics"].get("generation_time", 0) for data in answers.values() if data["regular"]["metrics"]]
        regular_tokens = [data["regular"]["metrics"].get("tokens", 0) for data in answers.values() if data["regular"]["metrics"]]
        
        mrl_search_times = [data["mrl"]["metrics"].get("search_time", 0) for data in answers.values() if data["mrl"]["metrics"]]
        mrl_gen_times = [data["mrl"]["metrics"].get("generation_time", 0) for data in answers.values() if data["mrl"]["metrics"]]
        mrl_tokens = [data["mrl"]["metrics"].get("tokens", 0) for data in answers.values() if data["mrl"]["metrics"]]
        
        avg_regular_search_time = np.mean(regular_search_times) if regular_search_times else "N/A"
        avg_regular_gen_time = np.mean(regular_gen_times) if regular_gen_times else "N/A"
        avg_regular_tokens = np.mean(regular_tokens) if regular_tokens else "N/A"
        
        avg_mrl_search_time = np.mean(mrl_search_times) if mrl_search_times else "N/A"
        avg_mrl_gen_time = np.mean(mrl_gen_times) if mrl_gen_times else "N/A"
        avg_mrl_tokens = np.mean(mrl_tokens) if mrl_tokens else "N/A"
        
        f.write("### Average Metrics\n\n")
        f.write("| Metric | Regular Embeddings | MRL Embeddings |\n")
        f.write("|--------|-------------------|---------------|\n")
        f.write(f"| Search Time | {avg_regular_search_time:.4f}s | {avg_mrl_search_time:.4f}s |\n")
        f.write(f"| Generation Time | {avg_regular_gen_time:.2f}s | {avg_mrl_gen_time:.2f}s |\n")
        f.write(f"| Tokens Used | {avg_regular_tokens:.1f} | {avg_mrl_tokens:.1f} |\n\n")
        
        # Speed improvement
        if avg_mrl_search_time != "N/A" and avg_regular_search_time != "N/A":
            search_speedup = (avg_regular_search_time / avg_mrl_search_time - 1) * 100
            f.write(f"**Search Speed Improvement with MRL:** {search_speedup:.2f}%\n\n")
        
        # Detailed results for each query
        f.write("## Detailed Results\n\n")
        
        for query, data in answers.items():
            f.write(f"### Query: \"{query}\"\n\n")
            
            # Regular embeddings
            f.write("#### Regular Embeddings\n\n")
            f.write("**Answer:**\n\n")
            f.write(f"```\n{data['regular']['answer']}\n```\n\n")
            
            if data["regular"]["metrics"]:
                search_time = data["regular"]["metrics"].get("search_time", "N/A")
                gen_time = data["regular"]["metrics"].get("generation_time", "N/A")
                tokens = data["regular"]["metrics"].get("tokens", "N/A")
                f.write(f"**Metrics:** Search Time: {search_time:.4f}s, Generation Time: {gen_time:.2f}s, Tokens: {tokens}\n\n")
            
            # MRL embeddings
            f.write(f"#### MRL Embeddings ({MRL_DIMENSION}d)\n\n")
            f.write("**Answer:**\n\n")
            f.write(f"```\n{data['mrl']['answer']}\n```\n\n")
            
            if data["mrl"]["metrics"]:
                search_time = data["mrl"]["metrics"].get("search_time", "N/A")
                gen_time = data["mrl"]["metrics"].get("generation_time", "N/A")
                tokens = data["mrl"]["metrics"].get("tokens", "N/A")
                f.write(f"**Metrics:** Search Time: {search_time:.4f}s, Generation Time: {gen_time:.2f}s, Tokens: {tokens}\n\n")
            
            f.write("---\n\n")
    
    logging.info(f"Markdown report generated at {report_path}")

if __name__ == "__main__":
    compare_with_deepseek()
