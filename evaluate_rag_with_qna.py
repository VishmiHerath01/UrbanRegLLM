import json
import csv
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import faiss
import numpy as np
from models.llm_clients import HuggingFaceClient

def load_resources():
    # Use a stronger model if possible
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Consider upgrading to 'paraphrase-mpnet-base-v2'
    index = faiss.read_index("data/embeddings/faiss_index.index")
    with open("data/chunks.txt", "r") as f:
        text_chunks = [line.strip() for line in f.readlines()]
    # Preprocess chunks for better quality
    text_chunks = preprocess_chunks(text_chunks)
    return model, index, text_chunks

def preprocess_chunks(chunks):
    """Clean and prepare chunks for better retrieval"""
    processed_chunks = []
    for chunk in chunks:
        # Remove chunk identifiers that might confuse the model
        chunk = re.sub(r'\[Chunk \d+\]', '', chunk)
        # Clean up whitespace
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        # Only include substantial chunks
        if len(chunk) > 100:  # Minimum content threshold
            processed_chunks.append(chunk)
    return processed_chunks

def get_context(question, model, index, text_chunks, top_k=5):  # Increased from 3 to 5
    # Apply query expansion to make retrieval more robust
    expanded_question = f"{question} urban regulations building development planning"
    
    # Encode with the model
    question_embedding = model.encode([expanded_question])[0].astype('float32')
    
    # Get more candidates initially, then filter
    k_candidates = min(top_k * 2, len(text_chunks))
    distances, indices = index.search(np.array([question_embedding]), k_candidates)
    
    # Filter by relevance score (cosine similarity threshold)
    relevant_chunks = [(text_chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    
    # Rerank the chunks by direct relevance to the original question
    reranked_chunks = rerank_chunks(question, [chunk for chunk, _ in relevant_chunks], model)
    
    # Take top-k after filtering and reranking
    reranked_chunks = reranked_chunks[:top_k]
    
    # Join the most relevant chunks
    return "\n\n".join(reranked_chunks)

def rerank_chunks(question, retrieved_chunks, model):
    """Re-rank chunks by direct relevance to question"""
    if not retrieved_chunks:
        return []
        
    question_embedding = model.encode([question])[0]
    chunk_embeddings = model.encode(retrieved_chunks)
    
    # Calculate similarities
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    
    # Create (chunk, similarity) pairs and sort
    chunk_scores = list(zip(retrieved_chunks, similarities))
    ranked_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    
    # Return sorted chunks
    return [chunk for chunk, _ in ranked_chunks]

def create_prompt(question, context):
    return f"""
You are an assistant specialized in Sri Lankan urban development regulations.
Answer the following question based on the context provided. 

IMPORTANT INSTRUCTIONS:
1. If the exact answer isn't in the context but you can reasonably infer it from the provided regulations, provide that inference.
2. If you mention regulation numbers or refer to specific rules, cite them directly from the context.
3. If the information is partially present, provide what you can find and note what's missing.
4. Only say "I don't have information about that in the regulations" if the context has absolutely no relevant information.
5. Keep your answer concise and focused on the question.

Context:
{context}

Question: {question}

Answer:
"""

def load_qna_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = []
    for section in data:
        for item in data[section]:
            questions.append((item['question'], item['answer']))
    return questions

def evaluate_rag_with_qna(json_file, output_file, hf_token):
    model, index, text_chunks = load_resources()
    client = HuggingFaceClient("mistralai/Mistral-7B-Instruct-v0.3", api_token=hf_token)
    qna_pairs = load_qna_from_json(json_file)

    results = []
    for q, gt in qna_pairs:
        print(f"Processing question: {q}")
        context = get_context(q, model, index, text_chunks)
        prompt = create_prompt(q, context)
        try:
            answer = client.generate(prompt)
        except Exception as e:
            answer = f"Error: {e}"

        results.append({
            "question": q,
            "ground_truth": gt,
            "retrieved_context": context,
            "generated_answer": answer
        })

    # Save results
    keys = results[0].keys()
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"Done! Results saved to {output_file}")

# Example usage
if __name__ == "__main__":
    json_file = "data/QnA Dataset/1st_batch.json"
    output_file = "data/rag_results-2.csv"
    hf_token = "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"

    evaluate_rag_with_qna(json_file, output_file, hf_token)
