"""Exception handlers for FastAPI application."""

import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

from src.memory_system.exceptions import MilvusConnectionError, LLMCallError

logger = logging.getLogger(__name__)


class APIError(HTTPException):
    """Base API error with consistent structure."""
    
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or f"ERR_{status_code}"


class DatabaseConnectionError(APIError):
    """Raised when database connection fails."""
    
    def __init__(self, detail: str = "Database connection failed"):
        super().__init__(status_code=503, detail=detail, error_code="DB_CONNECTION_ERROR")


class LLMServiceError(APIError):
    """Raised when LLM service fails."""
    
    def __init__(self, detail: str = "LLM service unavailable"):
        super().__init__(status_code=502, detail=detail, error_code="LLM_SERVICE_ERROR")


class MemoryNotFoundError(APIError):
    """Raised when requested memory is not found."""
    
    def __init__(self, memory_id: int):
        super().__init__(
            status_code=404,
            detail=f"Memory with id {memory_id} not found",
            error_code="MEMORY_NOT_FOUND"
        )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    
    @app.exception_handler(MilvusConnectionError)
    async def milvus_connection_handler(request: Request, exc: MilvusConnectionError):
        logger.error(f"Milvus connection error: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "error_code": "DB_CONNECTION_ERROR",
                "detail": f"Database connection failed: {exc.uri}",
                "message": str(exc)
            }
        )
    
    @app.exception_handler(LLMCallError)
    async def llm_call_handler(request: Request, exc: LLMCallError):
        logger.error(f"LLM call error: {exc}")
        return JSONResponse(
            status_code=502,
            content={
                "error_code": "LLM_SERVICE_ERROR",
                "detail": f"LLM service failed after {exc.attempts} attempts",
                "model": exc.model,
                "message": str(exc)
            }
        )
    
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code,
                "detail": exc.detail
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "detail": "An internal error occurred",
                "message": str(exc) if logger.isEnabledFor(logging.DEBUG) else None
            }
        )
