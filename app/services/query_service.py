from fastapi import HTTPException, Request
from app.schemas.query import QueryRequest
from app.services.ai_service import generate_answer

async def query_service(request: Request, query_data: QueryRequest):
    try:
        query_embedding = request.app.state.embedding_model.encode([query_data.query]).tolist()[0]
        results = request.app.state.collection.query(query_embeddings=[query_embedding], n_results=3)
        
        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])

        print(documents)
        
        if not documents or not documents[0]:
            return {"answer": "No relevant info found."}
        
        context = "\n\n".join(documents[0])
        answer = generate_answer(
            query_data.query, context, 
            request.app.state.tokenizer, request.app.state.llm_model
        )
        
        sources = list(set([m.get("filename", "Unknown") for m in metadatas[0]]))
        return {"query": query_data.query, "answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))