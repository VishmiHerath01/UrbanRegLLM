import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configuration
DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

def load_comparison_results():
    """Load the comparison results from the JSON files"""
    regular_results_path = os.path.join(RESULTS_DIR, "regular_results.json")
    mrl_results_path = os.path.join(RESULTS_DIR, "mrl_results_256d.json")  # Using 256d as recommended
    
    if not os.path.exists(regular_results_path) or not os.path.exists(mrl_results_path):
        print(f"Results files not found. Please run compare_embeddings.py first.")
        return None, None
    
    with open(regular_results_path, 'r') as f:
        regular_results = json.load(f)
    
    with open(mrl_results_path, 'r') as f:
        mrl_results = json.load(f)
    
    return regular_results, mrl_results

def visualize_retrieval_scores(regular_results, mrl_results):
    """Create visualizations comparing retrieval scores for each query"""
    if not regular_results or not mrl_results:
        return
    
    # Get the test queries
    test_queries = regular_results.get('test_queries', [])
    if not test_queries:
        print("No test queries found in results.")
        return
    
    # Create a summary visualization of all queries
    create_summary_visualization(regular_results, mrl_results, test_queries)
    
    # Create individual visualizations for each query
    for query in test_queries:
        create_query_visualization(query, regular_results, mrl_results)

def create_summary_visualization(regular_results, mrl_results, test_queries):
    """Create a summary visualization comparing average top-3 scores for all queries"""
    # Calculate average top-3 scores for each query
    regular_avg_scores = []
    mrl_avg_scores = []
    query_labels = []
    
    for i, query in enumerate(test_queries):
        if query in regular_results.get('retrieval_results', {}) and query in mrl_results.get('retrieval_results', {}):
            # Get top 3 scores for regular embeddings
            regular_scores = [result['score'] for result in regular_results['retrieval_results'][query][:3]]
            regular_avg = sum(regular_scores) / len(regular_scores) if regular_scores else 0
            
            # Get top 3 scores for MRL embeddings
            mrl_scores = [result['score'] for result in mrl_results['retrieval_results'][query][:3]]
            mrl_avg = sum(mrl_scores) / len(mrl_scores) if mrl_scores else 0
            
            regular_avg_scores.append(regular_avg)
            mrl_avg_scores.append(mrl_avg)
            
            # Create shortened query label
            short_query = query[:30] + "..." if len(query) > 30 else query
            query_labels.append(f"Q{i+1}: {short_query}")
    
    if not regular_avg_scores or not mrl_avg_scores:
        print("No scores found for visualization.")
        return
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(query_labels))
    width = 0.35
    
    plt.bar(x - width/2, regular_avg_scores, width, label='Regular Embeddings')
    plt.bar(x + width/2, mrl_avg_scores, width, label='MRL Embeddings (256d)')
    
    plt.xlabel('Queries')
    plt.ylabel('Average Score (Top 3 Results)')
    plt.title('Comparison of Average Retrieval Scores by Query')
    plt.xticks(x, query_labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'summary_scores.png'))
    plt.close()
    
    # Create a difference chart to highlight improvements
    plt.figure(figsize=(12, 8))
    
    score_diffs = [mrl - reg for mrl, reg in zip(mrl_avg_scores, regular_avg_scores)]
    colors = ['green' if diff > 0 else 'red' for diff in score_diffs]
    
    plt.bar(x, score_diffs, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.xlabel('Queries')
    plt.ylabel('Score Difference (MRL - Regular)')
    plt.title('MRL Improvement Over Regular Embeddings')
    plt.xticks(x, query_labels, rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'score_improvements.png'))
    plt.close()

def create_query_visualization(query, regular_results, mrl_results):
    """Create a visualization comparing scores for a specific query"""
    if query not in regular_results.get('retrieval_results', {}) or query not in mrl_results.get('retrieval_results', {}):
        return
    
    # Get scores for regular embeddings
    regular_scores = [result['score'] for result in regular_results['retrieval_results'][query][:5]]
    
    # Get scores for MRL embeddings
    mrl_scores = [result['score'] for result in mrl_results['retrieval_results'][query][:5]]
    
    # Ensure both lists have 5 elements
    regular_scores = regular_scores + [0] * (5 - len(regular_scores))
    mrl_scores = mrl_scores + [0] * (5 - len(mrl_scores))
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    
    x = np.arange(5)  # 5 results
    width = 0.35
    
    plt.bar(x - width/2, regular_scores, width, label='Regular Embeddings')
    plt.bar(x + width/2, mrl_scores, width, label='MRL Embeddings (256d)')
    
    plt.xlabel('Result Rank')
    plt.ylabel('Score')
    plt.title(f'Retrieval Scores for Query: "{query[:50]}..."' if len(query) > 50 else f'Retrieval Scores for Query: "{query}"')
    plt.xticks(x, [f"#{i+1}" for i in range(5)])
    plt.legend()
    
    # Add score values on top of bars
    for i, score in enumerate(regular_scores):
        plt.text(i - width/2, score + 0.01, f"{score:.2f}", ha='center', va='bottom', fontsize=9)
    
    for i, score in enumerate(mrl_scores):
        plt.text(i + width/2, score + 0.01, f"{score:.2f}", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Create a safe filename from the query
    safe_filename = "".join([c if c.isalnum() else "_" for c in query]).lower()
    safe_filename = safe_filename[:50]  # Limit filename length
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, f'query_{safe_filename}.png'))
    plt.close()

def create_dimension_comparison():
    """Create visualizations comparing different MRL dimensions"""
    # Load results for different dimensions
    dimensions = [768, 512, 256, 128, 64]
    dimension_results = {}
    
    for dim in dimensions:
        mrl_results_path = os.path.join(RESULTS_DIR, f"mrl_results_{dim}d.json")
        if os.path.exists(mrl_results_path):
            with open(mrl_results_path, 'r') as f:
                dimension_results[dim] = json.load(f)
    
    if not dimension_results:
        print("No dimension results found.")
        return
    
    # Get a common set of queries across all dimensions
    common_queries = set()
    for dim, results in dimension_results.items():
        if 'retrieval_results' in results:
            if not common_queries:
                common_queries = set(results['retrieval_results'].keys())
            else:
                common_queries &= set(results['retrieval_results'].keys())
    
    common_queries = list(common_queries)
    
    if not common_queries:
        print("No common queries found across dimensions.")
        return
    
    # Calculate average top-3 scores for each dimension
    dimension_avg_scores = {}
    for dim, results in dimension_results.items():
        avg_scores = []
        for query in common_queries:
            if query in results['retrieval_results']:
                scores = [result['score'] for result in results['retrieval_results'][query][:3]]
                avg = sum(scores) / len(scores) if scores else 0
                avg_scores.append(avg)
        
        if avg_scores:
            dimension_avg_scores[dim] = sum(avg_scores) / len(avg_scores)
    
    if not dimension_avg_scores:
        print("No scores found for dimension comparison.")
        return
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    
    dims = sorted(dimension_avg_scores.keys())
    scores = [dimension_avg_scores[dim] for dim in dims]
    
    plt.plot(dims, scores, marker='o', linestyle='-', linewidth=2)
    
    plt.xlabel('MRL Dimension')
    plt.ylabel('Average Score (Top 3 Results)')
    plt.title('Impact of MRL Dimension on Retrieval Performance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add score values next to points
    for dim, score in zip(dims, scores):
        plt.text(dim, score + 0.01, f"{score:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'dimension_comparison.png'))
    plt.close()

def create_memory_vs_performance():
    """Create a visualization comparing memory usage vs. performance across dimensions"""
    # Load results for different dimensions
    dimensions = [768, 512, 256, 128, 64]
    dimension_data = {}
    
    for dim in dimensions:
        mrl_results_path = os.path.join(RESULTS_DIR, f"mrl_results_{dim}d.json")
        if os.path.exists(mrl_results_path):
            with open(mrl_results_path, 'r') as f:
                results = json.load(f)
                
                # Get memory usage
                memory = results.get('memory_metrics', {}).get('size', 0)
                
                # Calculate average score
                avg_score = 0
                if 'retrieval_results' in results:
                    scores = []
                    for query, query_results in results['retrieval_results'].items():
                        query_scores = [result['score'] for result in query_results[:3]]
                        if query_scores:
                            scores.append(sum(query_scores) / len(query_scores))
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                
                dimension_data[dim] = {
                    'memory': memory,
                    'score': avg_score
                }
    
    if not dimension_data:
        print("No dimension data found.")
        return
    
    # Create the visualization
    plt.figure(figsize=(10, 6))
    
    dims = sorted(dimension_data.keys())
    memories = [dimension_data[dim]['memory'] for dim in dims]
    scores = [dimension_data[dim]['score'] for dim in dims]
    
    # Normalize scores to 0-1 range for better visualization
    min_score = min(scores)
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) if max_score > min_score else 0.5 for score in scores]
    
    # Create scatter plot with size proportional to dimension
    plt.scatter(memories, normalized_scores, s=[dim/2 for dim in dims], alpha=0.7)
    
    # Add dimension labels to points
    for dim, mem, score in zip(dims, memories, normalized_scores):
        plt.annotate(f"{dim}d", (mem, score), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Normalized Retrieval Score')
    plt.title('Memory Usage vs. Retrieval Performance by Dimension')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add a trend line
    if len(memories) > 1:
        z = np.polyfit(memories, normalized_scores, 1)
        p = np.poly1d(z)
        plt.plot(memories, p(memories), "r--", alpha=0.7)
    
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'memory_vs_performance.png'))
    plt.close()

def update_comparison_report():
    """Update the comparison report to include the new visualizations"""
    report_path = os.path.join(RESULTS_DIR, "comparison_report.md")
    
    if not os.path.exists(report_path):
        print(f"Comparison report not found at {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # Add visualizations section if it doesn't exist
    if "## Visualizations" not in report_content:
        visualization_section = """
## Visualizations

### Summary of Retrieval Scores

![Summary of Retrieval Scores](visualizations/summary_scores.png)

### MRL Improvement Over Regular Embeddings

![Score Improvements](visualizations/score_improvements.png)

### Impact of MRL Dimension on Retrieval Performance

![Dimension Comparison](visualizations/dimension_comparison.png)

### Memory Usage vs. Retrieval Performance

![Memory vs Performance](visualizations/memory_vs_performance.png)

### Individual Query Visualizations

The following visualizations show the retrieval scores for individual queries:

"""
        # Add individual query visualizations
        regular_results, _ = load_comparison_results()
        if regular_results and 'test_queries' in regular_results:
            for i, query in enumerate(regular_results['test_queries']):
                safe_filename = "".join([c if c.isalnum() else "_" for c in query]).lower()
                safe_filename = safe_filename[:50]
                visualization_section += f"#### Query {i+1}: {query}\n\n"
                visualization_section += f"![Query {i+1} Scores](visualizations/query_{safe_filename}.png)\n\n"
        
        # Append the visualization section to the report
        with open(report_path, 'a') as f:
            f.write(visualization_section)
        
        print(f"Updated comparison report with visualizations at {report_path}")

if __name__ == "__main__":
    # Load comparison results
    regular_results, mrl_results = load_comparison_results()
    
    if regular_results and mrl_results:
        # Create visualizations directory
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        
        # Create visualizations
        visualize_retrieval_scores(regular_results, mrl_results)
        create_dimension_comparison()
        create_memory_vs_performance()
        
        # Update the comparison report
        update_comparison_report()
        
        print(f"Visualizations created in {VISUALIZATIONS_DIR}")
    else:
        print("Could not load comparison results. Please run compare_embeddings.py first.")
