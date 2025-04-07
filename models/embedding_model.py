from sentence_transformers import SentenceTransformer
import numpy as np
import os

###Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# # # Load chunks from the file
# # chunks_file_path = '../data/chunks.txt'

# # with open(chunks_file_path, 'r', encoding='utf-8') as file:
# #     chunks = file.read().split('\n\n')  # Assuming each chunk is separated by two newlines

# # # Remove chunk labels like "[Chunk 1]" (optional)
# # chunks = [chunk.split('\n', 1)[1] if '\n' in chunk else chunk for chunk in chunks]

# # # Embed each chunk
# # embeddings = model.encode(chunks, convert_to_tensor=True)

# # # Convert embeddings to a numpy array 
# # embeddings_array = embeddings.cpu().detach().numpy()

# # # Ensure the directory exists before saving
# # embeddings_dir = '../data/embeddings/'
# # os.makedirs(embeddings_dir, exist_ok=True)

# # # Save the embeddings to a file
# # np.save(os.path.join(embeddings_dir, 'embeddings.npy'), embeddings_array)

# # print(f"Embedding complete! {len(chunks)} chunks embedded.")

# # import numpy as np

# # # Load the embeddings from the .npy file
# # embeddings_file_path = '../data/embeddings/embeddings.npy'
# # embeddings = np.load(embeddings_file_path)

# # # Check the shape of the embeddings (how many chunks and embedding dimensions)
# # print(f"Shape of embeddings: {embeddings.shape}")

# # # Print the embeddings of the first chunk (optional)
# # print(f"First embedding (for Chunk 1):\n{embeddings[0]}")

# # # Check the type of the embeddings (should be numpy arrays)
# # print(f"Type of embeddings: {type(embeddings)}")

import faiss
import numpy as np

# Load the embeddings from the .npy file
embeddings_file_path = '../data/embeddings/embeddings.npy'
embeddings = np.load(embeddings_file_path)

# Load the corresponding text chunks
chunks_file_path = '../data/chunks.txt'

with open(chunks_file_path, 'r', encoding='utf-8') as file:
    chunks = file.read().split('\n\n')  # Assuming each chunk is separated by two newlines

# Remove chunk labels like "[Chunk 1]" (optional)
chunks = [chunk.split('\n', 1)[1] if '\n' in chunk else chunk for chunk in chunks]

# Ensure that embeddings are in float32 (FAISS requirement)
embeddings = embeddings.astype(np.float32)

# Create a FAISS index for the embeddings
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance metric for similarity search

# Add the embeddings to the index
index.add(embeddings)

# Save the index to a file
faiss.write_index(index, '../data/embeddings/faiss_index.index')

print("FAISS index created and saved!")


def search_faiss_index(query, top_k=5):
    # Embed the query (using the same model you used for chunks)
    query_embedding = model.encode([query], convert_to_tensor=True).cpu().detach().numpy().astype(np.float32)

    # Search the FAISS index for the top_k most similar embeddings
    D, I = index.search(query_embedding, top_k)

    # Print the results
    print(f"Top {top_k} most similar chunks:")
    for i in range(top_k):
        print(f"\n{chunks[I[0][i]]}\n")
        print(f"Distance: {D[0][i]}")

# Example query
query = "What are the regulations for construction permits?"
search_faiss_index(query)
