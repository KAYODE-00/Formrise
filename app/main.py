from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
load_dotenv()

app = FastAPI()


@tool
def search_company_docs(query: str) -> str:
    """Search internal company policy documents for information about leave, shipping, or refunds."""
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    return "\n\n".join(results["documents"][0])

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="company_docs",
    embedding_function=embedding_fn
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class Question(BaseModel):
    text: str

@app.post("/ask")
def ask_question(q: Question):
    llm = ChatGroq(model="openai/gpt-oss-120b")
    llm_with_tools = llm.bind_tools([search_company_docs])

    response = llm_with_tools.invoke(q.text)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_result = search_company_docs.invoke(tool_call["args"])
        return {"question": q.text, "used_tool": True, "answer": tool_result}
    else:
        return {"question": q.text, "used_tool": False, "answer": response.content}