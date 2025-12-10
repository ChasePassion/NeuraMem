"""Memory management API endpoints."""

import asyncio
import logging
from fastapi import APIRouter, Depends, Path

from src.memory_system import Memory
from src.api.deps import get_memory_system
from src.api.schemas import (
    ManageRequest,
    ManageResponse,
    SearchRequest,
    SearchResponse,
    MemoryResponse,
    DeleteRequest,
    DeleteResponse,
    ResetRequest,
    ResetResponse,
    ConsolidateRequest,
    ConsolidateResponse,
)
from src.api.exceptions import MemoryNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


@router.post("/manage", response_model=ManageResponse)
async def manage_memories(
    request: ManageRequest,
    memory: Memory = Depends(get_memory_system)
) -> ManageResponse:
    """Manage memories with CRUD operations based on conversation.
    
    Processes user and assistant text through the memory manager
    to determine add, update, or delete operations.
    """
    try:
        added_ids = await memory.manage_async(
            user_text=request.user_text,
            assistant_text=request.assistant_text,
            user_id=request.user_id,
            chat_id=request.chat_id
        )
        
        logger.info(f"Manage completed for user {request.user_id}: added {len(added_ids)} memories")
        return ManageResponse(added_ids=added_ids, success=True)
    
    except Exception as e:
        logger.error(f"Manage failed for user {request.user_id}: {e}")
        raise


@router.post("/search", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest,
    memory: Memory = Depends(get_memory_system)
) -> SearchResponse:
    """Search memories for a user with narrative group expansion.
    
    Retrieves both episodic and semantic memories. Episodic memories
    are expanded to include all members of their narrative groups.
    """
    result = await asyncio.to_thread(
        memory.search,
        query=request.query,
        user_id=request.user_id
    )
    
    # Convert MemoryRecord objects to response models
    episodic = [
        MemoryResponse(
            id=m.id,
            user_id=m.user_id,
            memory_type=m.memory_type,
            ts=m.ts,
            chat_id=m.chat_id,
            text=m.text,
            group_id=m.group_id
        )
        for m in result.get("episodic", [])
    ]
    
    semantic = [
        MemoryResponse(
            id=m.id,
            user_id=m.user_id,
            memory_type=m.memory_type,
            ts=m.ts,
            chat_id=m.chat_id,
            text=m.text,
            group_id=m.group_id
        )
        for m in result.get("semantic", [])
    ]
    
    logger.info(
        f"Search for user {request.user_id}: "
        f"found {len(episodic)} episodic, {len(semantic)} semantic"
    )
    
    return SearchResponse(episodic=episodic, semantic=semantic)


@router.delete("/{memory_id}", response_model=DeleteResponse)
async def delete_memory(
    memory_id: int = Path(..., description="Memory ID to delete"),
    request: DeleteRequest = None,
    memory: Memory = Depends(get_memory_system)
) -> DeleteResponse:
    """Delete a single memory record.
    
    Requires user_id for ownership verification and narrative group cleanup.
    """
    user_id = request.user_id if request else None
    
    success = await asyncio.to_thread(
        memory.delete,
        memory_id=memory_id,
        user_id=user_id
    )
    
    if not success:
        raise MemoryNotFoundError(memory_id)
    
    logger.info(f"Deleted memory {memory_id} for user {user_id}")
    return DeleteResponse(success=True, deleted_count=1)


@router.delete("/reset", response_model=ResetResponse)
async def reset_memories(
    request: ResetRequest,
    memory: Memory = Depends(get_memory_system)
) -> ResetResponse:
    """Delete all memories for a user."""
    count = await asyncio.to_thread(
        memory.reset,
        user_id=request.user_id
    )
    
    logger.info(f"Reset completed for user {request.user_id}: deleted {count} memories")
    return ResetResponse(success=True, deleted_count=count)


@router.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate_memories(
    request: ConsolidateRequest,
    memory: Memory = Depends(get_memory_system)
) -> ConsolidateResponse:
    """Run memory consolidation process.
    
    Extracts semantic facts from episodic memories through
    batch pattern merging.
    """
    stats = await asyncio.to_thread(
        memory.consolidate,
        user_id=request.user_id
    )
    
    logger.info(
        f"Consolidation for user {request.user_id or 'all'}: "
        f"processed {stats.memories_processed}, created {stats.semantic_created}"
    )
    
    return ConsolidateResponse(
        memories_processed=stats.memories_processed,
        semantic_created=stats.semantic_created
    )
