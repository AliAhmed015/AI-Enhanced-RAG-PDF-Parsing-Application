from fastapi import Request

from app.schemas.query import QueryRequest

from app.services.query_service import query_service

async def query_controller(request: Request, query_data: QueryRequest):
    return await query_service(request=request, query_data=query_data)