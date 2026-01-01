from fastapi import File, Request, UploadFile


from app.services.document_service import document_service

async def document_controller(request: Request, file: UploadFile = File(...)):
    return await document_service(request=request, file=file)