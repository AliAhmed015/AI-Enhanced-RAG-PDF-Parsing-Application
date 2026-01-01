from fastapi import Request

from app.services.clear_service import clear_service

async def clear_controller(request: Request):
    return await clear_service(request=request)