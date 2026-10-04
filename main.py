from pathlib import Path

from fastapi import FastAPI, Query
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

KB_DIR = Path(__file__).resolve().parent / "knowledgebase"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

# -----------------------------
# 1. Load & Prepare Data (only first time)
# -----------------------------
documents = PyPDFLoader(str(KB_DIR / "Cricket-Rules.pdf")).load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

# -----------------------------
# 2. Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 3. Create or Load Chroma DB
# -----------------------------
if CHROMA_DIR.exists():
    # Load existing database
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    print("Loaded existing Chroma DB")
else:
    # Create new database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Created new Chroma DB and stored chunks")

# -----------------------------
# 4. Query Function
# -----------------------------
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
    
    return response

# -----------------------------
# 5. FastAPI App
# -----------------------------
app = FastAPI(title="Cricket RAG API")

@app.get("/")
async def root():
    return {"message": "Cricket RAG API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/query")
async def ask_question(q: str = Query(..., description="Your cricket question")):
    """
    Ask any question about Cricket Rules
    Example: /query?q=What is the length of the pitch?
    """
    return query_cricket(q)