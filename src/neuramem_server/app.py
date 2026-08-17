"""FastAPI application assembly.

Run: uvicorn neuramem_server.app:app (legacy entry was
uvicorn src.api.main:app — the REST contract itself is unchanged).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neuramem_server.deps import get_memory_system
from neuramem_server.exceptions import register_exception_handlers
from neuramem_server.routers import chat, memories
from neuramem_server.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # fail fast: construct the singleton before serving traffic
    get_memory_system()
    logger.info("NeuraMem server started")
    yield
    logger.info("NeuraMem server shutting down")


app = FastAPI(
    title="NeuraMem API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(chat.router)
app.include_router(memories.router)


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "NeuraMem API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "neuramem_server.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() in ("1", "true", "yes"),
    )
