import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from models.llm_clients import HuggingFaceClient


# Load saved embeddings and FAISS index
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    index = faiss.read_index("data/embeddings/faiss_index.index")
    with open("data/chunks.txt", "r") as f:
        text_chunks = [line.strip() for line in f]
    # Preprocess chunks for better quality
    text_chunks = preprocess_chunks(text_chunks)
    return model, index, text_chunks


# Process chunks to improve retrieval quality
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


# Retrieve relevant context with improved techniques
def get_context(question, model, index, text_chunks, top_k=5):  
    # Apply query expansion to make retrieval more robust
    expanded_question = f"{question} urban regulations building development planning"
    
    # Encode with the model
    question_embedding = model.encode([expanded_question])[0].astype('float32')
    
    # Get more candidates initially, then filter
    k_candidates = min(top_k * 2, len(text_chunks))
    distances, indices = index.search(np.array([question_embedding]), k_candidates)
    
    # Filter and combine with retrieved chunks
    relevant_chunks = [(text_chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    
    # Rerank chunks by direct relevance to the original question
    reranked_chunks = rerank_chunks(question, [chunk for chunk, _ in relevant_chunks], model)
    
    # Take top-k after reranking
    reranked_chunks = reranked_chunks[:top_k]
    
    # Join the most relevant chunks
    return "\n\n".join(reranked_chunks)


# Rerank chunks by relevance to the original question
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


# Create prompt for the LLM with improved instructions
def create_prompt(question, context):
    return f"""
You are an assistant specializing in Sri Lankan urban development regulations. 
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


# Initialize only the Hugging Face LLM client
def initialize_client(api_token, model_name):
    if not api_token:
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("Hugging Face API token is missing.")
    return HuggingFaceClient(model_name, api_token=api_token)


# Main pipeline: Retrieval + Generation with improved RAG
def main():
    # Configs
    hf_token = "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"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3" 
    

    # Load resources
    model, index, text_chunks = load_resources()

    # Initialize client
    hf_client = initialize_client(hf_token, model_name)

    # User input
    question = input("Please enter your question: ")

    # Step 1: Retrieve context with improved retrieval
    print(f"\n--- Retrieving context for: '{question}' ---")
    context = get_context(question, model, index, text_chunks)
    print("\n Retrieved Context:\n", context)

    # Step 2: Generate answer with improved prompt
    prompt = create_prompt(question, context)
    print("\n Generating answer...")
    answer = hf_client.generate(prompt)

    # Output
    print("\n Final Answer:\n", answer)
    print("\nPipeline complete (Retrieval + Generation)\n")


if __name__ == "__main__":
    main()
