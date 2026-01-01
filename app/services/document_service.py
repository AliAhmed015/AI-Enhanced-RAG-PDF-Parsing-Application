import os
import uuid
import pymupdf4llm
from fastapi import File, UploadFile, HTTPException, Request
from app.services.ai_service import chunk_text
from app.core.config import UPLOAD_DIR


async def document_service(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        markdown_content = pymupdf4llm.to_markdown(file_path)
        chunks = chunk_text(markdown_content)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text content found")
        
        # Access shared resources from app state
        embeddings = request.app.state.embedding_model.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]
        
        request.app.state.collection.add(
            documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids
        )
        
        return {"status": "success", "chunks_created": len(chunks), "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))