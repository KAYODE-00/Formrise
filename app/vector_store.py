import chromadb
from chromadb.utils import embedding_functions
from app.chunker import load_and_chunk

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="company_docs",
    embedding_function=embedding_fn
)

chunks = load_and_chunk("data/company_policy.txt")

collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Stored {collection.count()} chunks in Chroma.")