"""Pydantic schemas for FastAPI endpoints."""

from typing import List, Optional
from pydantic import BaseModel, Field


# ============== Chat Models ==============

class ChatMessage(BaseModel):
    """Single chat message in history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for SSE chat endpoint."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Conversation/thread identifier")
    message: str = Field(..., description="User message")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history")


class ChatChunkEvent(BaseModel):
    """SSE event for streaming chat chunk."""
    type: str = "chunk"
    content: str


class ChatDoneEvent(BaseModel):
    """SSE event for chat completion."""
    type: str = "done"
    full_content: str


# ============== Memory Models ==============

class ManageRequest(BaseModel):
    """Request model for manage endpoint."""
    user_id: str = Field(..., description="User identifier")
    chat_id: str = Field(..., description="Conversation/thread identifier")
    user_text: str = Field(..., description="User input text")
    assistant_text: str = Field(..., description="Assistant response text")


class ManageResponse(BaseModel):
    """Response model for manage endpoint."""
    added_ids: List[int] = Field(default_factory=list, description="IDs of newly added memories")
    success: bool = True


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    user_id: str = Field(..., description="User identifier")
    query: str = Field(..., description="Search query text")


class MemoryResponse(BaseModel):
    """Single memory record response."""
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    group_id: int = -1


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    episodic: List[MemoryResponse] = Field(default_factory=list)
    semantic: List[MemoryResponse] = Field(default_factory=list)


class DeleteRequest(BaseModel):
    """Request model for delete by user_id."""
    user_id: str = Field(..., description="User identifier for ownership verification")


class DeleteResponse(BaseModel):
    """Response model for delete operations."""
    success: bool
    deleted_count: int = 0


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""
    user_id: str = Field(..., description="User identifier")


class ResetResponse(BaseModel):
    """Response model for reset endpoint."""
    success: bool
    deleted_count: int


class ConsolidateRequest(BaseModel):
    """Request model for consolidate endpoint."""
    user_id: Optional[str] = Field(None, description="Optional user ID, consolidates all if None")


class ConsolidateResponse(BaseModel):
    """Response model for consolidate endpoint."""
    memories_processed: int
    semantic_created: int


# ============== Health Check ==============

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = "ok"
    version: str = "1.0.0"
