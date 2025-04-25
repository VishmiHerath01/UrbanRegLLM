import os
import json
import logging
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = "data"

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
        with open("test.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        # Extract questions and answers from the test data
        test_queries = []
        ground_truth = {}
        
        for item in test_data:
            if 'question' in item and 'answer' in item:
                test_queries.append(item['question'])
                ground_truth[item['question']] = item['answer']
        
        logging.info(f"Loaded {len(test_queries)} test queries from test.json")
        
        # If there are too many queries, select a representative sample
        if len(test_queries) > 10:
            # Select queries from different categories if available
            categories = {}
            for item in test_data:
                if 'category' in item and 'question' in item:
                    cat = item['category'].lower()
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(item['question'])
            
            # Select up to 2 queries from each category
            selected_queries = []
            for cat, queries in categories.items():
                selected_queries.extend(queries[:min(2, len(queries))])
            
            # If we still need more queries, add random ones
            import random
            if len(selected_queries) < 10:
                remaining = [q for q in test_queries if q not in selected_queries]
                selected_queries.extend(random.sample(remaining, min(10-len(selected_queries), len(remaining))))
            
            # Filter ground truth to match selected queries
            filtered_ground_truth = {q: ground_truth[q] for q in selected_queries if q in ground_truth}
            
            test_queries = selected_queries[:10]  # Limit to 10 queries for performance
            ground_truth = filtered_ground_truth
            logging.info(f"Selected {len(test_queries)} representative test queries")
        
        return test_queries, ground_truth
    except Exception as e:
        logging.error(f"Error loading test queries: {str(e)}")
        # Fallback to default queries
        return [
            "What are the requirements for natural light and ventilation in buildings?",
            "What are the minimum plot coverage requirements for buildings?",
            "How are blind walls and boundary structures regulated for safety?",
            "What are the rainwater harvesting requirements in different rainfall zones?",
            "What are the regulations for construction permits?"
        ], {}

# Load test queries and ground truth answers
TEST_QUERIES, GROUND_TRUTH = load_test_queries()

# Load your API key from an environment variable or configuration file
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=OPENAI_API_KEY)

def load_retrieval_results():
    """Load retrieval results from previous comparison"""
    regular_results_path = os.path.join(RESULTS_DIR, "regular_results.json")
    mrl_results_path = os.path.join(RESULTS_DIR, "mrl_results_256d.json")  # Using 256d as recommended
    
    if not os.path.exists(regular_results_path) or not os.path.exists(mrl_results_path):
        logging.error("Retrieval results not found. Run compare_embeddings.py first.")
        return None, None
    
    with open(regular_results_path, 'r') as f:
        regular_results = json.load(f)
    
    with open(mrl_results_path, 'r') as f:
        mrl_results = json.load(f)
    
    return regular_results, mrl_results

def generate_answer(query, context, model="mistral"):
    """Generate an answer using a language model"""
    if model == "mistral":
        # This is a placeholder for Mistral integration
        # You would need to implement the actual Mistral API call here
        # For now, we'll simulate it with a delay
        time.sleep(1)
        
        # Simulated answer
        return f"Based on the provided context, here's an answer to '{query}'..."
    
    elif model == "openai":
        # Uncomment and use this if you have OpenAI API access
        # prompt = f"""
        # Answer the following question based only on the provided context:
        # 
        # Context:
        # {context}
        # 
        # Question: {query}
        # 
        # Answer:
        # """
        # 
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=0.3,
        #     max_tokens=500
        # )
        # 
        # return response.choices[0].message.content
        
        # Simulated answer for now
        return f"Based on the provided context, here's an answer to '{query}'..."
    
    else:
        raise ValueError(f"Unsupported model: {model}")

def evaluate_answer_quality(regular_results, mrl_results, model="mistral"):
    """Evaluate the quality of answers generated with both embedding approaches"""
    quality_results = {
        "regular": {},
        "mrl": {},
        "ground_truth": GROUND_TRUTH,
        "test_queries": TEST_QUERIES
    }
    
    for query in TEST_QUERIES:
        logging.info(f"Evaluating answer quality for query: {query}")
        
        # Get context from regular embeddings
        if query in regular_results.get('retrieval_results', {}):
            regular_context = "\n\n".join([
                f"Document {i+1}:\n{result['content']}" 
                for i, result in enumerate(regular_results['retrieval_results'][query][:3])
            ])
            
            # Generate answer with regular embeddings
            regular_answer = generate_answer(query, regular_context, model)
            quality_results["regular"][query] = {
                "context": regular_context,
                "answer": regular_answer
            }
        
        # Get context from MRL embeddings
        if query in mrl_results.get('retrieval_results', {}):
            mrl_context = "\n\n".join([
                f"Document {i+1}:\n{result['content']}" 
                for i, result in enumerate(mrl_results['retrieval_results'][query][:3])
            ])
            
            # Generate answer with MRL embeddings
            mrl_answer = generate_answer(query, mrl_context, model)
            quality_results["mrl"][query] = {
                "context": mrl_context,
                "answer": mrl_answer
            }
    
    # Save results
    with open(os.path.join(RESULTS_DIR, f"answer_quality_{model}.json"), 'w') as f:
        json.dump(quality_results, f, indent=2)
    
    return quality_results

def generate_human_evaluation_form(quality_results):
    """Generate a form for human evaluation of answer quality"""
    form_path = os.path.join(RESULTS_DIR, "human_evaluation_form.md")
    
    with open(form_path, 'w') as f:
        f.write("# Answer Quality Evaluation Form\n\n")
        f.write("For each query, please rate the quality of the answers on a scale of 1-5:\n")
        f.write("1: Poor, 2: Fair, 3: Good, 4: Very Good, 5: Excellent\n\n")
        
        # List the test queries used
        f.write("## Test Queries Used\n\n")
        if 'test_queries' in quality_results:
            for i, query in enumerate(quality_results['test_queries']):
                f.write(f"{i+1}. {query}\n")
        f.write("\n")
        
        for i, query in enumerate(TEST_QUERIES):
            f.write(f"## Query {i+1}: {query}\n\n")
            
            # Answer A and B will be randomly assigned to regular or MRL
            # to avoid bias in evaluation
            use_regular_first = np.random.choice([True, False])
            
            if use_regular_first:
                answer_a = quality_results["regular"].get(query, {}).get("answer", "No answer generated")
                answer_b = quality_results["mrl"].get(query, {}).get("answer", "No answer generated")
                answer_a_type = "regular"
                answer_b_type = "mrl"
            else:
                answer_a = quality_results["mrl"].get(query, {}).get("answer", "No answer generated")
                answer_b = quality_results["regular"].get(query, {}).get("answer", "No answer generated")
                answer_a_type = "mrl"
                answer_b_type = "regular"
            
            f.write("### Answer A:\n\n")
            f.write(f"```\n{answer_a}\n```\n\n")
            f.write("Rating (1-5): ____\n\n")
            
            f.write("### Answer B:\n\n")
            f.write(f"```\n{answer_b}\n```\n\n")
            f.write("Rating (1-5): ____\n\n")
            
            f.write("### Comments (optional):\n\n")
            f.write("____________________\n\n")
            
            # Save the mapping for later analysis
            f.write(f"<!-- Answer A: {answer_a_type}, Answer B: {answer_b_type} -->\n\n")
    
    logging.info(f"Human evaluation form generated at {form_path}")

def generate_answer_quality_report(quality_results):
    """Generate a report comparing the answer quality"""
    report_path = os.path.join(RESULTS_DIR, "answer_quality_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Answer Quality Comparison Report\n\n")
        
        # List the test queries used
        f.write("## Test Queries Used\n\n")
        if 'test_queries' in quality_results:
            for i, query in enumerate(quality_results['test_queries']):
                f.write(f"{i+1}. {query}\n")
        f.write("\n")
        
        for query in TEST_QUERIES:
            f.write(f"## Query: {query}\n\n")
            
            # Regular embeddings answer
            f.write("### Answer with Regular Embeddings\n\n")
            regular_answer = quality_results["regular"].get(query, {}).get("answer", "No answer generated")
            f.write(f"```\n{regular_answer}\n```\n\n")
            
            # MRL embeddings answer
            f.write("### Answer with MRL Embeddings (256d)\n\n")
            mrl_answer = quality_results["mrl"].get(query, {}).get("answer", "No answer generated")
            f.write(f"```\n{mrl_answer}\n```\n\n")
            
            # Ground truth answer (if available)
            if query in quality_results.get("ground_truth", {}):
                f.write("### Ground Truth Answer\n\n")
                ground_truth = quality_results["ground_truth"][query]
                f.write(f"```\n{ground_truth}\n```\n\n")
            
            # Context comparison
            f.write("### Context Comparison\n\n")
            
            regular_context = quality_results["regular"].get(query, {}).get("context", "")
            mrl_context = quality_results["mrl"].get(query, {}).get("context", "")
            
            # Check if contexts are different
            if regular_context != mrl_context:
                f.write("The contexts used for generating these answers were different, which may explain differences in answer quality.\n\n")
            else:
                f.write("The contexts used for generating these answers were identical.\n\n")
    
    logging.info(f"Answer quality report generated at {report_path}")

def run_answer_quality_evaluation(model="mistral"):
    """Run the answer quality evaluation"""
    # Load retrieval results
    regular_results, mrl_results = load_retrieval_results()
    
    if regular_results is None or mrl_results is None:
        return
    
    # Evaluate answer quality
    quality_results = evaluate_answer_quality(regular_results, mrl_results, model)
    
    # Generate human evaluation form
    generate_human_evaluation_form(quality_results)
    
    # Generate answer quality report
    generate_answer_quality_report(quality_results)

if __name__ == "__main__":
    run_answer_quality_evaluation(model="mistral")
