import os
import shutil
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pypdf import PdfReader

load_dotenv()

app = FastAPI()

os.makedirs("data", exist_ok=True)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="company_docs", embedding_function=embedding_fn
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@tool
def search_company_docs(query: str) -> str:
    """Search internal company policy documents for information about leave, shipping, or refunds."""
    results = collection.query(query_texts=[query], n_results=2)
    return "\n".join(results["documents"][0])


class Question(BaseModel):
    text: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted

    print(f"Total extracted text length: {len(full_text)}")
    print(f"First 300 characters: {full_text[:300]}")

    chunks = full_text.split("\n\n")
    chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 50]

    if not chunks:
        return {"filename": file.filename, "chunks_added": 0, "message": "No valid text chunks found."}

    collection.add(
        documents=chunks, 
        ids=[f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
    )

    return {"filename": file.filename, "chunks_added": len(chunks)}


@app.post("/ask")
def ask_question(q: Question):
    llm = ChatGroq(model="openai/gpt-oss-120b")
    llm_with_tools = llm.bind_tools([search_company_docs])

    response = llm_with_tools.invoke(q.text)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        search_query = tool_call["args"].get("query", q.text)
        tool_result = search_company_docs.invoke({"query": search_query})
        context = tool_result

        prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know."

Context: {context}
Question: {q.text}
"""

        groq_response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b", 
            messages=[{"role": "user", "content": prompt}]
        )

        answer = groq_response.choices[0].message.content
        return {"question": q.text, "used_tool": True, "answer": answer}
        
    else:
        return {"question": q.text, "used_tool": False, "answer": response.content}
