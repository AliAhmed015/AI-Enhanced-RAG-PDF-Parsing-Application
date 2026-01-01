from fastapi import Request


async def clear_service(request: Request):
    request.app.state.chroma_client.delete_collection(name="documents")
    request.app.state.collection = request.app.state.chroma_client.create_collection(name="documents")
    return {"status": "success"}