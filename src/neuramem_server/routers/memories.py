"""Memory management REST endpoints — contract unchanged from legacy.

Delete no longer pre-checks ownership through private store access: the
facade's delete_async already scopes by user_id and reports whether a
row was actually removed.
"""

import logging

from fastapi import APIRouter, Depends, Path, Query

from neuramem.memory import Memory
from neuramem_server.deps import get_memory_system
from neuramem_server.exceptions import MemoryNotFoundError
from neuramem_server.schemas import (
    ConsolidateRequest,
    ConsolidateResponse,
    DeleteResponse,
    ManageRequest,
    ManageResponse,
    MemoryResponse,
    ResetRequest,
    ResetResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


def _to_response(record) -> MemoryResponse:
    return MemoryResponse(
        id=record.id,
        user_id=record.user_id,
        memory_type=record.memory_type,
        ts=record.ts,
        chat_id=record.chat_id,
        text=record.text,
        group_id=record.group_id,
    )


@router.post("/manage", response_model=ManageResponse)
async def manage_memories(
    request: ManageRequest,
    memory: Memory = Depends(get_memory_system),
) -> ManageResponse:
    added_ids = await memory.manage_async(
        user_text=request.user_text,
        assistant_text=request.assistant_text,
        user_id=request.user_id,
        chat_id=request.chat_id,
    )
    return ManageResponse(added_ids=added_ids)


@router.post("/search", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest,
    memory: Memory = Depends(get_memory_system),
) -> SearchResponse:
    result = await memory.search_async(request.query, request.user_id)
    return SearchResponse(
        episodic=[_to_response(r) for r in result.episodic],
        semantic=[_to_response(r) for r in result.semantic],
    )


@router.delete("/reset", response_model=ResetResponse)
async def reset_memories(
    request: ResetRequest,
    memory: Memory = Depends(get_memory_system),
) -> ResetResponse:
    deleted = await memory.reset_async(request.user_id)
    return ResetResponse(success=True, deleted_count=deleted)


@router.delete("/{memory_id}", response_model=DeleteResponse)
async def delete_memory(
    memory_id: int = Path(...),
    user_id: str = Query(...),
    memory: Memory = Depends(get_memory_system),
) -> DeleteResponse:
    deleted = await memory.delete_async(memory_id, user_id)
    if not deleted:
        raise MemoryNotFoundError(memory_id)
    return DeleteResponse(success=True, deleted_count=1)


@router.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate_memories(
    request: ConsolidateRequest,
    memory: Memory = Depends(get_memory_system),
) -> ConsolidateResponse:
    stats = await memory.consolidate_async(request.user_id)
    return ConsolidateResponse(
        memories_processed=stats.memories_processed,
        semantic_created=stats.semantic_created,
    )
