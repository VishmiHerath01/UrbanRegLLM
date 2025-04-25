import os
import json
import csv
import re

# Configuration
DATA_DIR = "data"
RESULTS_DIR = os.path.join(DATA_DIR, "comparison_results")

def extract_retrieval_results():
    """Extract retrieval results from the comparison report and save to CSV"""
    # Check if comparison report exists
    report_path = os.path.join(RESULTS_DIR, "comparison_report.md")
    if not os.path.exists(report_path):
        print(f"Comparison report not found at {report_path}")
        return
    
    # Load the report content
    with open(report_path, 'r') as f:
        report_content = f.read()
    
    # Extract queries and results
    queries = []
    results = {}
    
    # Extract queries
    query_section = re.search(r'## Test Queries Used\n\n(.*?)\n\n##', report_content, re.DOTALL)
    if query_section:
        query_lines = query_section.group(1).strip().split('\n')
        for line in query_lines:
            if line.strip():
                # Extract query from numbered list (e.g., "1. What are...")
                match = re.match(r'\d+\.\s+(.*)', line)
                if match:
                    queries.append(match.group(1))
    
    # Extract results for each query
    for query in queries:
        # Create a safe version of the query for regex
        safe_query = re.escape(query)
        
        # Find the section for this query
        query_section = re.search(f'### Query: "{safe_query}"(.*?)(?=### Query:|$)', report_content, re.DOTALL)
        if not query_section:
            continue
        
        section_content = query_section.group(1)
        
        # Extract regular embeddings results
        regular_section = re.search(r'#### Regular Embeddings\n\n(.*?)(?=####|$)', section_content, re.DOTALL)
        regular_results = []
        
        if regular_section:
            regular_content = regular_section.group(1)
            # Extract individual results
            result_pattern = r'\d+\.\s+\*\*Score:\s+([\d\.]+)\*\*\s+Source:\s+(.*?)\s+Excerpt:\s+(.*?)(?=\d+\.\s+\*\*Score:|$)'
            for match in re.finditer(result_pattern, regular_content, re.DOTALL):
                score = match.group(1)
                source = match.group(2).strip()
                excerpt = match.group(3).strip().replace('\n', ' ')
                regular_results.append({
                    'score': score,
                    'source': source,
                    'excerpt': excerpt
                })
        
        # Extract MRL embeddings results
        mrl_section = re.search(r'#### MRL Embeddings \(256d\)\n\n(.*?)(?=###|$)', section_content, re.DOTALL)
        mrl_results = []
        
        if mrl_section:
            mrl_content = mrl_section.group(1)
            # Extract individual results
            result_pattern = r'\d+\.\s+\*\*Score:\s+([\d\.]+)\*\*\s+Source:\s+(.*?)\s+Excerpt:\s+(.*?)(?=\d+\.\s+\*\*Score:|$)'
            for match in re.finditer(result_pattern, mrl_content, re.DOTALL):
                score = match.group(1)
                source = match.group(2).strip()
                excerpt = match.group(3).strip().replace('\n', ' ')
                mrl_results.append({
                    'score': score,
                    'source': source,
                    'excerpt': excerpt
                })
        
        # Store results for this query
        results[query] = {
            'regular': regular_results,
            'mrl': mrl_results
        }
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "retrieval_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Query', 
            'Embedding Type', 
            'Result Rank', 
            'Score', 
            'Source', 
            'Retrieved Text'
        ])
        
        # Write data
        for query, query_results in results.items():
            # Write regular embedding results
            for i, result in enumerate(query_results['regular']):
                writer.writerow([
                    query if i == 0 else '',  # Only write query for first result
                    'Regular' if i == 0 else '',  # Only write type for first result
                    i + 1,
                    result['score'],
                    result['source'],
                    result['excerpt']
                ])
            
            # Write MRL embedding results
            for i, result in enumerate(query_results['mrl']):
                writer.writerow([
                    query if i == 0 else '',  # Only write query for first result
                    'MRL (256d)' if i == 0 else '',  # Only write type for first result
                    i + 1,
                    result['score'],
                    result['source'],
                    result['excerpt']
                ])
    
    print(f"Retrieval comparison saved to {csv_path}")
    
    # Also create a simplified version with just top results
    simplified_csv_path = os.path.join(RESULTS_DIR, "retrieval_comparison_simplified.csv")
    with open(simplified_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Query', 
            'Regular Top Result Score', 
            'Regular Top Result Text',
            'MRL Top Result Score',
            'MRL Top Result Text'
        ])
        
        # Write data
        for query, query_results in results.items():
            regular_top = query_results['regular'][0] if query_results['regular'] else {'score': 'N/A', 'excerpt': 'N/A'}
            mrl_top = query_results['mrl'][0] if query_results['mrl'] else {'score': 'N/A', 'excerpt': 'N/A'}
            
            writer.writerow([
                query,
                regular_top['score'],
                regular_top['excerpt'],
                mrl_top['score'],
                mrl_top['excerpt']
            ])
    
    print(f"Simplified retrieval comparison saved to {simplified_csv_path}")

if __name__ == "__main__":
    extract_retrieval_results()
