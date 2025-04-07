from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load your saved embeddings and FAISS index
def load_resources():
    # Load your sentence transformer model (same one used for document embedding)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load your FAISS index
    index = faiss.read_index("data/embeddings/faiss_index.index")
    
    # Load your text chunks (assuming you saved these separately)
    # This could be from a pickle file, JSON, etc.
    with open("data/chunks.txt", "rb") as f:
        import pickle
        text_chunks = pickle.load(f)
    
    return model, index, text_chunks

# 2. Function to get relevant context for a question
def get_context(question, model, index, text_chunks, top_k=3):
    # Embed the question using the same model
    question_embedding = model.encode([question])[0]
    
    # Convert to the format FAISS expects
    question_embedding = np.array([question_embedding]).astype('float32')
    
    # Search the index
    distances, indices = index.search(question_embedding, top_k)
    
    # Get the corresponding text chunks
    context_chunks = [text_chunks[idx] for idx in indices[0]]
    
    # Join the context chunks
    context = "\n\n".join(context_chunks)
    
    return context

# 3. Create the prompt with context
def create_prompt(question, context):
    prompt = f"""
You are an assistant specializing in Sri Lankan urban development regulations. 
Answer the question based ONLY on the following context. If the information isn't 
in the context, say "I don't have information about that in the regulations."

Context:
{context}

Question: {question}

Answer:
"""
    return prompt

# 4. Full RAG pipeline function
def answer_question(question, llm_client):
    # Load resources
    model, index, text_chunks = load_resources()
    
    # Get relevant context
    context = get_context(question, model, index, text_chunks)
    
    # Create prompt with context
    prompt = create_prompt(question, context)
    
    # Get answer from LLM
    # This depends on which LLM API you're using (OpenAI, local model, etc.)
    response = llm_client.generate(prompt)
    
    return response