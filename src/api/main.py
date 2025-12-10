"""FastAPI application entry point for NeuraMem API."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import chat, memories
from src.api.exceptions import register_exception_handlers
from src.api.schemas import HealthResponse
from src.api.deps import get_memory_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info("Starting NeuraMem API server...")
    
    # Pre-initialize memory system to fail fast if config is wrong
    try:
        memory = get_memory_system()
        logger.info(f"Memory system ready: {memory.config.collection_name}")
    except Exception as e:
        logger.error(f"Failed to initialize memory system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down NeuraMem API server...")


# Create FastAPI application
app = FastAPI(
    title="NeuraMem API",
    description="AI Memory System with episodic and semantic memory management",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
register_exception_handlers(app)

# Include routers
app.include_router(chat.router)
app.include_router(memories.router)


@app.get("/v1/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "NeuraMem API",
        "docs": "/docs",
        "health": "/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
