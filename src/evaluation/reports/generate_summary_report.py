import os
import json
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data")
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_all_results():
    """Load all results from the comparison_results directory"""
    results = {
        'metrics': {},
        'quality': {},
        'answers': {}
    }
    
    # Load metrics
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    if os.path.exists(metrics_dir):
        for filename in os.listdir(metrics_dir):
            if filename.endswith('_summary.json'):
                model_name = filename.split('_')[0]
                file_path = os.path.join(metrics_dir, filename)
                
                with open(file_path, 'r') as f:
                    results['metrics'][model_name] = json.load(f)
    
    # Load quality metrics
    quality_dir = os.path.join(RESULTS_DIR, "quality")
    if os.path.exists(quality_dir):
        for filename in os.listdir(quality_dir):
            if filename.endswith('_quality_summary.json'):
                model_name = filename.split('_')[0]
                file_path = os.path.join(quality_dir, filename)
                
                with open(file_path, 'r') as f:
                    results['quality'][model_name] = json.load(f)
    
    # Load raw answers
    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith('_answers.json'):
            model_name = filename.split('_')[0]
            file_path = os.path.join(RESULTS_DIR, filename)
            
            with open(file_path, 'r') as f:
                results['answers'][model_name] = json.load(f)
    
    return results

def generate_performance_summary(results):
    """Generate a performance summary table"""
    if not results or 'metrics' not in results or not results['metrics']:
        return "No performance metrics available."
    
    # Prepare data for the table
    data = []
    
    for model, metrics in results['metrics'].items():
        row = {'Model': model}
        
        # Add regular metrics
        if 'regular' in metrics:
            row['Search Time (s)'] = metrics['regular'].get('avg_search_time', 'N/A')
            row['Generation Time (s)'] = metrics['regular'].get('avg_generation_time', 'N/A')
            row['Tokens'] = metrics['regular'].get('avg_tokens', 'N/A')
        
        # Add MRL metrics if available
        if 'mrl' in metrics:
            row['MRL Search Time (s)'] = metrics['mrl'].get('avg_search_time', 'N/A')
            row['MRL Generation Time (s)'] = metrics['mrl'].get('avg_generation_time', 'N/A')
            row['MRL Tokens'] = metrics['mrl'].get('avg_tokens', 'N/A')
            row['MRL Dimension'] = metrics['mrl'].get('avg_dimension', 256)
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format the table
    table = df.to_markdown(index=False)
    return table

def generate_quality_summary(results):
    """Generate a quality summary table"""
    if not results or 'quality' not in results or not results['quality']:
        return "No quality metrics available."
    
    # Prepare data for the table
    data = []
    
    for model, metrics in results['quality'].items():
        row = {'Model': model}
        
        # Add regular metrics
        if 'regular' in metrics:
            row['Word Count'] = metrics['regular'].get('avg_word_count', 'N/A')
            row['Factual Consistency'] = metrics['regular'].get('avg_factual_consistency', 'N/A')
            row['Uncertainty (%)'] = metrics['regular'].get('pct_has_uncertainty_markers', 'N/A')
            row['Citations (%)'] = metrics['regular'].get('pct_has_citations', 'N/A')
        
        # Add MRL metrics if available
        if 'mrl' in metrics:
            row['MRL Word Count'] = metrics['mrl'].get('avg_word_count', 'N/A')
            row['MRL Factual Consistency'] = metrics['mrl'].get('avg_factual_consistency', 'N/A')
            row['MRL Uncertainty (%)'] = metrics['mrl'].get('pct_has_uncertainty_markers', 'N/A')
            row['MRL Citations (%)'] = metrics['mrl'].get('pct_has_citations', 'N/A')
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format the table
    table = df.to_markdown(index=False)
    return table

def generate_mrl_dimension_summary(results):
    """Generate a summary of MRL dimensions performance"""
    if not results or 'metrics' not in results or not results['metrics']:
        return "No MRL dimension metrics available."
    
    # Check if we have MRL data with different dimensions
    mrl_dimensions = {}
    
    for model, data in results['metrics'].items():
        if 'mrl' in data and 'dimension' in data['mrl']:
            dimension = data['mrl']['dimension']
            
            if dimension not in mrl_dimensions:
                mrl_dimensions[dimension] = {
                    'search_times': [],
                    'generation_times': [],
                    'tokens': []
                }
            
            if 'avg_search_time' in data['mrl']:
                mrl_dimensions[dimension]['search_times'].append(data['mrl']['avg_search_time'])
            
            if 'avg_generation_time' in data['mrl']:
                mrl_dimensions[dimension]['generation_times'].append(data['mrl']['avg_generation_time'])
            
            if 'avg_tokens' in data['mrl']:
                mrl_dimensions[dimension]['tokens'].append(data['mrl']['avg_tokens'])
    
    if not mrl_dimensions:
        return "No MRL dimension data available."
    
    # Prepare data for the table
    data = []
    
    for dimension, metrics in sorted(mrl_dimensions.items()):
        row = {'Dimension': dimension}
        
        if metrics['search_times']:
            row['Avg Search Time (s)'] = np.mean(metrics['search_times'])
        else:
            row['Avg Search Time (s)'] = 'N/A'
        
        if metrics['generation_times']:
            row['Avg Generation Time (s)'] = np.mean(metrics['generation_times'])
        else:
            row['Avg Generation Time (s)'] = 'N/A'
        
        if metrics['tokens']:
            row['Avg Tokens'] = np.mean(metrics['tokens'])
        else:
            row['Avg Tokens'] = 'N/A'
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Format the table
    table = df.to_markdown(index=False)
    return table

def generate_sample_comparison(results):
    """Generate a sample comparison of answers"""
    if not results or 'answers' not in results or not results['answers']:
        return "No answers available for comparison."
    
    # Get a random query that exists in all models
    common_queries = set()
    first_model = True
    
    for model, answers in results['answers'].items():
        if first_model:
            common_queries = set(answers.keys())
            first_model = False
        else:
            common_queries = common_queries.intersection(set(answers.keys()))
    
    if not common_queries:
        return "No common queries found across models."
    
    # Select a random query
    import random
    sample_query = random.choice(list(common_queries))
    
    # Generate the comparison
    comparison = f"## Sample Comparison for Query: \"{sample_query}\"\n\n"
    
    for model, answers in results['answers'].items():
        comparison += f"### {model.upper()}\n\n"
        
        if 'regular' in answers[sample_query]:
            comparison += "**Regular Embeddings:**\n\n"
            comparison += f"```\n{answers[sample_query]['regular']['answer']}\n```\n\n"
        
        if 'mrl' in answers[sample_query]:
            comparison += f"**MRL Embeddings ({answers[sample_query]['mrl'].get('dimension', 256)}d):**\n\n"
            comparison += f"```\n{answers[sample_query]['mrl']['answer']}\n```\n\n"
    
    return comparison

def generate_summary_report():
    """Generate a comprehensive summary report"""
    # Load all results
    results = load_all_results()
    
    # Generate the report
    report = "# Legal Embedding Comparison Summary Report\n\n"
    report += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    report += "## Overview\n\n"
    report += "This report summarizes the performance and quality metrics for different embedding approaches "
    report += "used in legal document retrieval and question answering.\n\n"
    
    report += "### Approaches Compared\n\n"
    report += "1. **Regular Embeddings**: Full-dimensional legal domain embeddings\n"
    report += "2. **MRL Embeddings**: Matryoshka Representation Learning embeddings with multiple dimensions\n\n"
    
    report += "### Models Used\n\n"
    for model in results.get('answers', {}).keys():
        report += f"- **{model.upper()}**\n"
    report += "\n"
    
    # Performance summary
    report += "## Performance Metrics\n\n"
    report += generate_performance_summary(results)
    report += "\n\n"
    
    # Quality summary
    report += "## Quality Metrics\n\n"
    report += generate_quality_summary(results)
    report += "\n\n"
    
    # MRL dimension summary
    report += "## MRL Dimension Comparison\n\n"
    report += generate_mrl_dimension_summary(results)
    report += "\n\n"
    
    # Sample comparison
    report += generate_sample_comparison(results)
    report += "\n\n"
    
    # Key findings
    report += "## Key Findings\n\n"
    
    # Search time comparison
    if 'metrics' in results and results['metrics']:
        regular_search_times = []
        mrl_search_times = []
        
        for model, metrics in results['metrics'].items():
            if 'regular' in metrics and 'avg_search_time' in metrics['regular']:
                regular_search_times.append(metrics['regular']['avg_search_time'])
            
            if 'mrl' in metrics and 'avg_search_time' in metrics['mrl']:
                mrl_search_times.append(metrics['mrl']['avg_search_time'])
        
        if regular_search_times and mrl_search_times:
            avg_regular = np.mean(regular_search_times)
            avg_mrl = np.mean(mrl_search_times)
            
            if avg_mrl < avg_regular:
                speedup = (avg_regular / avg_mrl - 1) * 100
                report += f"- **MRL Embeddings are {speedup:.2f}% faster** for search compared to regular embeddings\n"
            else:
                slowdown = (avg_mrl / avg_regular - 1) * 100
                report += f"- MRL Embeddings are {slowdown:.2f}% slower for search compared to regular embeddings\n"
    
    # Quality comparison
    if 'quality' in results and results['quality']:
        regular_factual = []
        mrl_factual = []
        
        for model, metrics in results['quality'].items():
            if 'regular' in metrics and 'avg_factual_consistency' in metrics['regular']:
                regular_factual.append(metrics['regular']['avg_factual_consistency'])
            
            if 'mrl' in metrics and 'avg_factual_consistency' in metrics['mrl']:
                mrl_factual.append(metrics['mrl']['avg_factual_consistency'])
        
        if regular_factual and mrl_factual:
            avg_regular = np.mean(regular_factual)
            avg_mrl = np.mean(mrl_factual)
            
            if avg_mrl > avg_regular:
                improvement = (avg_mrl / avg_regular - 1) * 100
                report += f"- **MRL Embeddings show {improvement:.2f}% better factual consistency** compared to regular embeddings\n"
            else:
                decline = (avg_regular / avg_mrl - 1) * 100
                report += f"- MRL Embeddings show {decline:.2f}% lower factual consistency compared to regular embeddings\n"
    
    # Save the report
    report_file = os.path.join(REPORTS_DIR, "summary_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    logging.info(f"Summary report generated and saved to {report_file}")
    return report_file

if __name__ == "__main__":
    generate_summary_report()
