def load_and_chunk(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = text.split("TOPIC")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) >= 50]
    return chunks


chunks = load_and_chunk("data/company_policy.txt")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print()

print(f"Total chunks: {len(chunks)}")