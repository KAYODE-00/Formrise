import chromadb
from app.chunker import load_and_chunk

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("company_docs")

chunks = load_and_chunk("data/company_policy.txt")


collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Stored {collection.count()} chunks in Chroma.")






