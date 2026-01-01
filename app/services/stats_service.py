from fastapi import Request

async def stats_service(request: Request):
    count = request.app.state.collection.count()
    all_items = request.app.state.collection.get()
    filenames = list(set([m.get("filename", "Unknown") for m in all_items.get("metadatas", [])]))
    return {"total_chunks": count, "documents": filenames}