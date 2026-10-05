from pathlib import Path

from fastapi import FastAPI, Query,Form
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KB_DIR = Path(__file__).resolve().parent / "knowledgebase"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"


documents = PyPDFLoader(str(KB_DIR / "Cricket-Rules.pdf")).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if CHROMA_DIR.exists():
    # Load existing database
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    print("Loaded existing Chroma DB")
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Created new Chroma DB and stored chunks")


def query_cricket(question: str, k: int = 3):
    """Search relevant chunks for a question"""
    results = vectorstore.similarity_search(question, k=k)
    
    response = {
        "question": question,
        "results": []
    }
    
    for i, doc in enumerate(results, 1):
        response["results"].append({
            "rank": i,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown")
        })
    print(response)
    return response


app = FastAPI(title="Cricket RAG API")

@app.get("/")
async def root():
    return {"message": "Cricket RAG API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/query")
async def ask_question(q: str = Form(..., description="Your cricket question")):
    """
    Ask any question about Cricket Rules
    Send as form-data with key: q
    """
    return query_cricket(q)