"""Domain exceptions -> HTTP responses (same mapping as the legacy layer)."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from neuramem.core.exceptions import LLMCallError, MilvusConnectionError

logger = logging.getLogger(__name__)


class APIError(HTTPException):
    """Base API error carrying a machine-readable error_code."""

    def __init__(self, status_code: int, detail: str, error_code: str = ""):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or f"ERR_{status_code}"


class DatabaseConnectionError(APIError):
    def __init__(self, detail: str = "Vector store unavailable"):
        super().__init__(503, detail, "DB_CONNECTION_ERROR")


class LLMServiceError(APIError):
    def __init__(self, detail: str = "LLM service failed"):
        super().__init__(502, detail, "LLM_SERVICE_ERROR")


class MemoryNotFoundError(APIError):
    def __init__(self, memory_id: int):
        super().__init__(404, f"Memory {memory_id} not found", "MEMORY_NOT_FOUND")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MilvusConnectionError)
    async def _milvus_error(request: Request, exc: MilvusConnectionError):
        return JSONResponse(
            status_code=503,
            content={
                "error_code": "DB_CONNECTION_ERROR",
                "detail": str(exc),
                "message": "Vector store connection failed",
            },
        )

    @app.exception_handler(LLMCallError)
    async def _llm_error(request: Request, exc: LLMCallError):
        return JSONResponse(
            status_code=502,
            content={
                "error_code": "LLM_SERVICE_ERROR",
                "detail": str(exc),
                "message": "LLM call failed",
                "attempts": exc.attempts,
                "model": exc.model,
            },
        )

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": exc.error_code, "detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error_code": "INTERNAL_ERROR", "detail": "Internal server error"},
        )
