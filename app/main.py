import chromadb
from fastapi import FastAPI
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from app.api.endpoints import router as api_router
from app.core.config import CHROMA_PATH

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load Models ---
    print("🚀 Loading Models...")
    app.state.embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    model_name = "gpt2"
    app.state.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    app.state.llm_model = GPT2LMHeadModel.from_pretrained(model_name)
    app.state.tokenizer.pad_token = app.state.tokenizer.eos_token
    
    app.state.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    app.state.collection = app.state.chroma_client.get_or_create_collection(name="documents")
    
    print("✅ System Ready!")
    yield
    # --- Shutdown: Cleanup ---
    print("🔄 Shutting down...")

app = FastAPI(title="AI Enhanced PDF Parser", lifespan=lifespan)

# Include the routes
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)