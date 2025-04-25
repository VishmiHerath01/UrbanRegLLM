#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import contextlib
import io


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


from rag_pipelines.enhanced.enhanced_rag_pipeline_mrl import enhanced_rag_pipeline, load_vector_db
import rag_pipelines.enhanced.enhanced_rag_pipeline_mrl


original_load_vector_db = load_vector_db


original_load_mistral_model = rag_pipelines.enhanced.enhanced_rag_pipeline_mrl.load_mistral_model


def patched_load_vector_db(dimension=256, index_path="mrl_legal_faiss_index"):

    correct_index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    "data/indices/mrl_legal_faiss_index")
    

    return original_load_vector_db(dimension=dimension, index_path=correct_index_path)


class SuppressOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout

def patched_load_mistral_model():
    with SuppressOutput():
        return original_load_mistral_model()


rag_pipelines.enhanced.enhanced_rag_pipeline_mrl.load_mistral_model = patched_load_mistral_model
rag_pipelines.enhanced.enhanced_rag_pipeline_mrl.load_vector_db = patched_load_vector_db


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S',
                   level=logging.INFO)

def print_header():
    print("\n" + "="*80)
    print(" Urban Development Regulations Assistant ".center(80, "="))
    print(" Enhanced RAG Pipeline with Multi-Stage Reasoning ".center(80, "="))
    print("="*80 + "\n")

def print_instructions():
    print("This assistant uses an enhanced RAG pipeline with:")
    print("1. MRL embeddings for precise document retrieval")
    print("2. DeepSeek-R1 for legal reasoning")
    print("3. Mistral-7B for final answer generation\n")
    print("Commands:")
    print("- Type your question about urban development regulations")
    print("- Type 'dimension X' to change MRL dimension (e.g., 'dimension 128')")
    print("- Type 'save' to save the last result to a file")
    print("- Type 'exit' to quit\n")

def interactive_mode(dimension=256):
    print_header()
    print_instructions()
    
    current_dimension = dimension
    last_result = None
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() == 'exit':
            print("\nThank you for using the Urban Development Regulations Assistant.")
            break
        
        # Handle dimension change command
        if query.lower().startswith('dimension '):
            try:
                new_dim = int(query.split()[1])
                if new_dim in [64, 128, 256, 512, 768]:
                    current_dimension = new_dim
                    print(f"\nChanged MRL dimension to {current_dimension}d")
                else:
                    print("\nSupported dimensions: 64, 128, 256, 512, 768")
                continue
            except (IndexError, ValueError):
                print("\nInvalid dimension format. Use 'dimension X' where X is a number.")
                continue
        
        # Handle save command
        if query.lower() == 'save' and last_result:
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"result_{int(time.time())}.txt")
            
            with open(save_path, 'w') as f:
                f.write(f"Query: {last_result['query']}\n\n")
                f.write("LEGAL REASONING:\n")
                f.write(f"{last_result['reasoning']}\n\n")
                f.write("FINAL ANSWER:\n")
                f.write(f"{last_result['answer']}\n\n")
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"Retrieval dimension: {last_result['dimension']}d\n")
                if last_result.get('info'):
                    f.write(f"Reasoning tokens: {last_result['info'].get('reasoning_tokens', 0)}\n")
                    f.write(f"Answer tokens: {last_result['info'].get('answer_tokens', 0)}\n")
                    f.write(f"Reasoning time: {last_result['info'].get('reasoning_time', 0):.2f}s\n")
                f.write(f"Total processing time: {last_result['total_time']:.2f}s\n")
            
            print(f"\nSaved result to {save_path}")
            continue
        
        # Process the query
        print(f"\nProcessing your question using MRL {current_dimension}d embeddings...")
        start_time = time.time()
        
        try:
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
            print("LEGAL REASONING")
            print("="*80)
            print(reasoning)
            
            if info:
                print(f"\nReasoning tokens: {info.get('reasoning_tokens', 0)}, Time: {info.get('reasoning_time', 0):.2f}s")
            
            # Display the final answer
            print("\n" + "="*80)
            print("FINAL ANSWER")
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
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            print(f"\nError: {str(e)}")

def single_query_mode(query, dimension=256):
    print_header()
    print(f"Processing query: {query}\n")
    
    start_time = time.time()
    
    try:
        answer, context, reasoning, info = enhanced_rag_pipeline(query, dimension=dimension)
        total_time = time.time() - start_time
        
        # Display the reasoning
        print("\n" + "="*80)
        print("LEGAL REASONING")
        print("="*80)
        print(reasoning)
        
        if info:
            print(f"\nReasoning tokens: {info.get('reasoning_tokens', 0)}, Time: {info.get('reasoning_time', 0):.2f}s")
        
        # Display the final answer
        print("\n" + "="*80)
        print("FINAL ANSWER")
        print("="*80)
        print(answer)
        
        # Display performance metrics
        print("\n" + "-"*80)
        print("PERFORMANCE METRICS")
        print("-"*80)
        if info:
            print(f"Retrieval dimension: {dimension}d")
            print(f"Reasoning tokens: {info.get('reasoning_tokens', 0)}")
            print(f"Answer tokens: {info.get('answer_tokens', 0)}")
            print(f"Reasoning time: {info.get('reasoning_time', 0):.2f}s")
            print(f"Total processing time: {total_time:.2f}s")
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Urban Development Regulations Assistant")
    
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--dimension", type=int, choices=[64, 128, 256, 512, 768], 
                      default=256, help="Dimension for MRL embeddings")
    
    args = parser.parse_args()
    

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_results")
    os.makedirs(save_dir, exist_ok=True)
    
    if args.query:

        single_query_mode(args.query, dimension=args.dimension)
    else:

        interactive_mode(dimension=args.dimension)

if __name__ == "__main__":
    main()
