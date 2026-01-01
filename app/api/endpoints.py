import os
import uuid
import pymupdf4llm
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from app.schemas.query import QueryRequest
from app.services.ai_service import chunk_text, generate_answer
from app.core.config import UPLOAD_DIR

from app.controllers.document_controller import document_controller
from app.controllers.query_controller import query_controller
from app.controllers.stats_controller import stats_controller
from app.controllers.clear_controller import clear_controller

router = APIRouter()

@router.get("/")
async def root():
    return {"status": "online", "message": "AI Enhanced PDF Parser API"}

@router.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    # if not file.filename.endswith('.pdf'):
    #     raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # try:
    #     file_path = os.path.join(UPLOAD_DIR, file.filename)
    #     content = await file.read()
    #     with open(file_path, "wb") as f:
    #         f.write(content)
        
    #     markdown_content = pymupdf4llm.to_markdown(file_path)
    #     chunks = chunk_text(markdown_content)
        
    #     if not chunks:
    #         raise HTTPException(status_code=400, detail="No text content found")
        
    #     # Access shared resources from app state
    #     embeddings = request.app.state.embedding_model.encode(chunks).tolist()
    #     ids = [str(uuid.uuid4()) for _ in chunks]
    #     metadatas = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]
        
    #     request.app.state.collection.add(
    #         documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids
    #     )
        
    #     return {"status": "success", "chunks_created": len(chunks), "filename": file.filename}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    return await document_controller(request=request, file=file)

@router.post("/query")
async def query_documents(request: Request, query_data: QueryRequest):
    # try:
    #     query_embedding = request.app.state.embedding_model.encode([query_data.query]).tolist()[0]
    #     results = request.app.state.collection.query(query_embeddings=[query_embedding], n_results=3)
        
    #     documents = results.get("documents", [[]])
    #     metadatas = results.get("metadatas", [[]])
        
    #     if not documents or not documents[0]:
    #         return {"answer": "No relevant info found."}
        
    #     context = "\n\n".join(documents[0])
    #     answer = generate_answer(
    #         query_data.query, context, 
    #         request.app.state.tokenizer, request.app.state.llm_model
    #     )
        
    #     sources = list(set([m.get("filename", "Unknown") for m in metadatas[0]]))
    #     return {"query": query_data.query, "answer": answer, "sources": sources}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

    return await query_controller(request=request, query_data=query_data)


@router.get("/stats")
async def get_stats(request: Request):
    # count = request.app.state.collection.count()
    # all_items = request.app.state.collection.get()
    # filenames = list(set([m.get("filename", "Unknown") for m in all_items.get("metadatas", [])]))
    # return {"total_chunks": count, "documents": filenames}
    return await stats_controller(request=request)


@router.delete("/clear")
async def clear_collection(request: Request):
    # request.app.state.chroma_client.delete_collection(name="documents")
    # request.app.state.collection = request.app.state.chroma_client.create_collection(name="documents")
    # return {"status": "success"}
    return await clear_controller(request=request)