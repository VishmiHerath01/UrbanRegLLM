import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import os
import re
from transformers import AutoTokenizer
from models.llm_clients import HuggingFaceClient


def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index("data/embeddings/faiss_index.index")
    with open("data/chunks.txt", "r") as f:
        text_chunks = [line.strip() for line in f]
    text_chunks = preprocess_chunks(text_chunks)
    return model, index, text_chunks


def preprocess_chunks(chunks):
    processed_chunks = []
    for chunk in chunks:
        chunk = re.sub(r'\[Chunk \d+\]', '', chunk)
        chunk = re.sub(r'\s+', ' ', chunk).strip()
        if len(chunk) > 100:
            processed_chunks.append(chunk)
    return processed_chunks


def get_context(question, model, index, text_chunks, top_k=5):
    expanded_question = f"{question} urban regulations building development planning"
    question_embedding = model.encode([expanded_question])[0].astype('float32')
    k_candidates = min(top_k * 2, len(text_chunks))
    distances, indices = index.search(np.array([question_embedding]), k_candidates)
    relevant_chunks = [(text_chunks[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    reranked_chunks = rerank_chunks(question, [chunk for chunk, _ in relevant_chunks], model)
    reranked_chunks = reranked_chunks[:top_k]
    return "\n\n".join(reranked_chunks)


def rerank_chunks(question, retrieved_chunks, model):
    if not retrieved_chunks:
        return []
    question_embedding = model.encode([question])[0]
    chunk_embeddings = model.encode(retrieved_chunks)
    similarities = util.pytorch_cos_sim(question_embedding, chunk_embeddings)[0]
    chunk_scores = list(zip(retrieved_chunks, similarities))
    ranked_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked_chunks]


def create_prompt(question, context, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_total_tokens=2048, max_new_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    instruction = ("You are an assistant specializing in Sri Lankan urban development regulations.\n"
                   "Answer the following question based on the context provided.\n\n"
                   "IMPORTANT INSTRUCTIONS:\n"
                   "1. If the exact answer isn't in the context but you can reasonably infer it from the provided regulations, provide that inference.\n"
                   "2. If you mention regulation numbers or refer to specific rules, cite them directly from the context.\n"
                   "3. If the information is partially present, provide what you can find and note what's missing.\n"
                   "4. Only say \"I don't have information about that in the regulations\" if the context has absolutely no relevant information.\n"
                   "5. Keep your answer concise and focused on the question.\n\n"
                   "Context:\n")

    footer = f"\n\nQuestion: {question}\n\nAnswer:"


    static_tokens = tokenizer.encode(instruction + footer)
    available_tokens = max_total_tokens - max_new_tokens - len(static_tokens)


    context_tokens = tokenizer.encode(context)
    if len(context_tokens) > available_tokens:
        context_tokens = context_tokens[:available_tokens]
    truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)

    return f"{instruction}{truncated_context}{footer}"


def initialize_client(api_token, model_name):
    if not api_token:
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("Hugging Face API token is missing.")
    return HuggingFaceClient(model_name, api_token=api_token)


def generate_response(question):
    hf_token = "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"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model, index, text_chunks = load_resources()
    hf_client = initialize_client(hf_token, model_name)
    context = get_context(question, model, index, text_chunks)
    prompt = create_prompt(question, context, model_name=model_name, max_total_tokens=2048, max_new_tokens=100)
    answer = hf_client.generate(prompt, max_new_tokens=100)
    return answer

def main():
    question = input("Please enter your question: ")
    answer = generate_response(question)
    print("\n Final Answer:\n", answer)
    print("\nPipeline complete (Retrieval + Generation)\n")

if __name__ == "__main__":
    main()
