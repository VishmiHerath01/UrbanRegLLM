import os


with open("../data/cleaned.txt", "r", encoding="utf-8") as f:
    full_text = f.read()


full_text = full_text.replace('\xa0', ' ').replace('\n', ' ')
full_text = ' '.join(full_text.split())  # Remove excess spaces


chunk_size = 400  
overlap = 50       


words = full_text.split()
chunks = []

i = 0
while i < len(words):
    chunk = words[i:i + chunk_size]
    chunk_text = ' '.join(chunk)
    chunks.append(chunk_text)
    i += chunk_size - overlap  


with open("../data/chunks.txt", "w", encoding="utf-8") as f:
    for idx, chunk in enumerate(chunks):
        f.write(f"[Chunk {idx+1}]\n{chunk}\n\n")

print(f"Created {len(chunks)} chunks (each ~{chunk_size} words).")
