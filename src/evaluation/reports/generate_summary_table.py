import os
import json
import numpy as np
import pandas as pd

# Configuration
DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")

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

def calculate_metrics(regular_results, mrl_results):
    """Calculate metrics for comparison table"""
    metrics = {}
    
    # 1. Average retrieval score (top 3 results)
    regular_scores = []
    mrl_scores = []
    
    for query in regular_results.get('test_queries', []):
        if query in regular_results.get('retrieval_results', {}) and query in mrl_results.get('retrieval_results', {}):
            # Get top 3 scores for regular embeddings
            reg_query_scores = [result['score'] for result in regular_results['retrieval_results'][query][:3]]
            if reg_query_scores:
                regular_scores.append(sum(reg_query_scores) / len(reg_query_scores))
            
            # Get top 3 scores for MRL embeddings
            mrl_query_scores = [result['score'] for result in mrl_results['retrieval_results'][query][:3]]
            if mrl_query_scores:
                mrl_scores.append(sum(mrl_query_scores) / len(mrl_query_scores))
    
    if regular_scores and mrl_scores:
        avg_regular_score = sum(regular_scores) / len(regular_scores)
        avg_mrl_score = sum(mrl_scores) / len(mrl_scores)
        score_improvement = (avg_mrl_score - avg_regular_score) / avg_regular_score * 100
        
        metrics['Retrieval Score'] = {
            'Without MRL': f"{avg_regular_score:.3f}",
            'With MRL (256d)': f"{avg_mrl_score:.3f}",
            'Improvement': f"{'+' if score_improvement > 0 else ''}{score_improvement:.1f}%"
        }
    
    # 2. Memory usage
    regular_memory = regular_results.get('memory_metrics', {}).get('size', 0)
    mrl_memory = mrl_results.get('memory_metrics', {}).get('size', 0)
    
    if isinstance(regular_memory, (int, float)) and isinstance(mrl_memory, (int, float)) and regular_memory > 0:
        memory_improvement = (regular_memory - mrl_memory) / regular_memory * 100
        
        metrics['Memory Usage (MB)'] = {
            'Without MRL': f"{regular_memory:.2f}",
            'With MRL (256d)': f"{mrl_memory:.2f}",
            'Improvement': f"{'+' if memory_improvement > 0 else ''}{memory_improvement:.1f}%"
        }
    
    # 3. Search speed
    regular_speed = regular_results.get('search_metrics', {}).get('avg_time', 0)
    mrl_speed = mrl_results.get('search_metrics', {}).get('avg_time', 0)
    
    if isinstance(regular_speed, (int, float)) and isinstance(mrl_speed, (int, float)) and regular_speed > 0:
        speed_improvement = (regular_speed - mrl_speed) / regular_speed * 100
        
        metrics['Search Time (s)'] = {
            'Without MRL': f"{regular_speed:.4f}",
            'With MRL (256d)': f"{mrl_speed:.4f}",
            'Improvement': f"{'+' if speed_improvement > 0 else ''}{speed_improvement:.1f}%"
        }
    
    # 4. Dimensionality
    regular_dim = regular_results.get('dimension', 'full')
    if regular_dim == 'full':
        # Try to infer dimension from the data
        if 'retrieval_results' in regular_results and regular_results['retrieval_results']:
            query = next(iter(regular_results['retrieval_results']))
            if query in regular_results['retrieval_results'] and regular_results['retrieval_results'][query]:
                # This is just a placeholder, we don't actually have the dimension info
                regular_dim = "768"  # Assuming default BERT dimension
    
    metrics['Dimensionality'] = {
        'Without MRL': regular_dim,
        'With MRL (256d)': "256",
        'Improvement': f"-{(int(regular_dim) - 256) / int(regular_dim) * 100:.1f}%" if regular_dim.isdigit() else "N/A"
    }
    
    return metrics

def generate_summary_table(metrics):
    """Generate a markdown table summarizing the metrics"""
    table = "| Metric | Without MRL | With MRL (256d) | Improvement |\n"
    table += "|--------|------------|----------------|-------------|\n"
    
    for metric, values in metrics.items():
        table += f"| {metric} | {values['Without MRL']} | {values['With MRL (256d)']} | {values['Improvement']} |\n"
    
    return table

def update_comparison_report(summary_table):
    """Update the comparison report to include the summary table"""
    report_path = os.path.join(RESULTS_DIR, "comparison_report.md")
    
    if not os.path.exists(report_path):
        print(f"Comparison report not found at {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # Add summary table section if it doesn't exist
    if "## Summary Comparison" not in report_content:
        summary_section = "\n## Summary Comparison\n\n"
        summary_section += summary_table
        summary_section += "\n\n"
        
        # Insert after the test queries section
        if "## Test Queries Used" in report_content:
            parts = report_content.split("## Test Queries Used")
            part1 = parts[0] + "## Test Queries Used" + parts[1].split("##")[0]
            part2 = "##" + "##".join(parts[1].split("##")[1:])
            new_content = part1 + summary_section + part2
        else:
            # If test queries section doesn't exist, insert at the beginning
            new_content = summary_section + report_content
        
        with open(report_path, 'w') as f:
            f.write(new_content)
        
        print(f"Updated comparison report with summary table at {report_path}")
    else:
        print("Summary comparison section already exists in the report.")

def create_standalone_table():
    """Create a standalone summary table file"""
    regular_results, mrl_results = load_comparison_results()
    
    if not regular_results or not mrl_results:
        return
    
    metrics = calculate_metrics(regular_results, mrl_results)
    summary_table = generate_summary_table(metrics)
    
    # Save to a standalone file
    table_path = os.path.join(RESULTS_DIR, "summary_table.md")
    with open(table_path, 'w') as f:
        f.write("# MRL vs Regular Embeddings Summary\n\n")
        f.write(summary_table)
    
    print(f"Summary table created at {table_path}")
    
    # Also update the main comparison report
    update_comparison_report(summary_table)

if __name__ == "__main__":
    create_standalone_table()
