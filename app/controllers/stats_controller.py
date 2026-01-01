from fastapi import Request

from app.services.stats_service import stats_service

async def stats_controller(request: Request):
    return await stats_service(request=request)