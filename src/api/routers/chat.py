"""SSE Chat endpoint with streaming response."""

import asyncio
import json
import logging
import time
from typing import List, Dict, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.memory_system import Memory, MemoryRecord
from src.memory_system.prompts import MEMORY_ANSWER_PROMPT
from src.api.deps import get_memory_system
from src.api.schemas import ChatRequest, ChatMessage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _build_context_with_memories(
    message: str,
    memories: Dict[str, List[MemoryRecord]],
    history: List[ChatMessage]
) -> str:
    """Build context string including memories and conversation history."""
    context_parts = []
    
    # 1. Episodic memories
    context_parts.append("Here are the episodic memories:")
    episodic_memories = memories.get("episodic", [])
    if episodic_memories:
        for i, mem in enumerate(episodic_memories[:5], 1):
            context_parts.append(f"{i}. {mem.text}")
    else:
        context_parts.append("(No episodic memories)")
    context_parts.append("")
    
    # 2. Semantic memories
    context_parts.append("Here are the semantic memories:")
    semantic_memories = memories.get("semantic", [])
    if semantic_memories:
        for i, mem in enumerate(semantic_memories[:5], 1):
            context_parts.append(f"{i}. {mem.text}")
    else:
        context_parts.append("(No semantic memories)")
    context_parts.append("")
    
    # 3. Conversation history
    context_parts.append("Here are the history messages:")
    if history:
        for i, msg in enumerate(history[-6:], 1):
            context_parts.append(f"  {msg.role}: {msg.content}")
    else:
        context_parts.append("(No history messages)")
    context_parts.append("")
    
    # 4. Current task
    context_parts.append("Here is the current user message:")
    context_parts.append(message)
    
    return "\n".join(context_parts)


@router.post("/v1/chat")
async def chat_stream(
    request: ChatRequest,
    memory: Memory = Depends(get_memory_system)
) -> StreamingResponse:
    """SSE streaming chat endpoint with memory-augmented responses.
    
    Flow:
    1. Search relevant memories for user
    2. Build context with memories and history
    3. Stream LLM response via SSE
    4. Async trigger memory management after completion
    
    SSE Event Format:
    - data: {"type": "chunk", "content": "..."} - Streaming token
    - data: {"type": "done", "full_content": "..."} - Completion event
    - data: {"type": "error", "message": "..."} - Error event
    """
    
    async def event_generator():
        accumulated_response = ""
        
        try:
            # 1. Search relevant memories
            relevant_memories = await asyncio.to_thread(
                memory.search,
                query=request.message,
                user_id=request.user_id
            )
            
            logger.info(
                f"Chat for user {request.user_id}: "
                f"found {len(relevant_memories.get('episodic', []))} episodic, "
                f"{len(relevant_memories.get('semantic', []))} semantic memories"
            )
            
            # 2. Build context
            context = _build_context_with_memories(
                request.message,
                relevant_memories,
                request.history
            )
            
            # Build system prompt
            system_prompt = f"{MEMORY_ANSWER_PROMPT}\n\nUser ID: {request.user_id}\n\n{context}"
            
            # 3. Stream LLM response
            async for chunk in memory._llm_client.chat_stream_async(
                system_prompt,
                request.message
            ):
                accumulated_response += chunk
                event_data = json.dumps({"type": "chunk", "content": chunk})
                yield f"data: {event_data}\n\n"
            
            # 4. Send completion event
            done_event = json.dumps({
                "type": "done",
                "full_content": accumulated_response
            })
            yield f"data: {done_event}\n\n"
            
            # 5. Async trigger memory management (fire-and-forget)
            asyncio.create_task(
                _manage_memory_background(
                    memory=memory,
                    user_message=request.message,
                    assistant_message=accumulated_response,
                    user_id=request.user_id,
                    chat_id=request.chat_id
                )
            )
            
        except Exception as e:
            logger.error(f"Chat stream error for user {request.user_id}: {e}")
            error_event = json.dumps({
                "type": "error",
                "message": str(e)
            })
            yield f"data: {error_event}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


async def _manage_memory_background(
    memory: Memory,
    user_message: str,
    assistant_message: str,
    user_id: str,
    chat_id: str
) -> None:
    """Background task to manage memory after chat completion."""
    try:
        added_ids = await memory.manage_async(
            user_text=user_message,
            assistant_text=assistant_message,
            user_id=user_id,
            chat_id=chat_id
        )
        logger.info(
            f"Background memory manage for user {user_id}: "
            f"added {len(added_ids)} memories"
        )
    except Exception as e:
        logger.error(f"Background memory manage failed: {e}")
