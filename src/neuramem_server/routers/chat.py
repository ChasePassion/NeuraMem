"""SSE chat endpoint — the two-phase closed loop's server consumer.

Flow (architecture_target.md ch. 11):
1. search_async -> SearchResult (correlation token)
2. server-owned LLM streams the answer (Memory is never pried open)
3. on done, fire-and-forget: report_usage_async (closes the loop — the
   legacy server NEVER did this, memories stayed group_id=-1 forever)
   + manage_async for the turn
"""

import asyncio
import json
import logging
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from neuramem.core.ports import LLM
from neuramem.memory import Memory
from neuramem.prompts import MEMORY_ANSWER_PROMPT
from neuramem_server.deps import get_chat_llm, get_memory_system
from neuramem_server.schemas import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# legacy context shape kept: memory blocks truncated to 5/5 explicitly
# (render defaults to no truncation — the retrieval config owns quantity)
MAX_EPISODIC_IN_PROMPT = 5
MAX_SEMANTIC_IN_PROMPT = 5
HISTORY_TURNS_IN_PROMPT = 6


def _build_history_section(history: List[ChatMessage]) -> str:
    parts = ["Here are the history messages:"]
    if history:
        parts.extend(
            f"  {msg.role}: {msg.content}" for msg in history[-HISTORY_TURNS_IN_PROMPT:]
        )
    else:
        parts.append("(No history messages)")
    parts.append("")
    parts.append("Here is the current user message:")
    return "\n".join(parts)


@router.post("/v1/chat")
async def chat_stream(
    request: ChatRequest,
    memory: Memory = Depends(get_memory_system),
    chat_llm: LLM = Depends(get_chat_llm),
) -> StreamingResponse:
    """SSE streaming chat with memory-augmented responses.

    Event format (unchanged from the legacy API):
    - data: {"type": "chunk", "content": "..."}
    - data: {"type": "done", "full_content": "..."}
    - data: {"type": "error", "message": "..."}
    """
    async def event_generator():
        accumulated = ""
        result = None
        try:
            # Phase 1: retrieval (correlation token for the write-back)
            result = await memory.search_async(request.message, request.user_id)
            logger.info(
                "Chat for user %s: %d episodic, %d semantic memories",
                request.user_id, len(result.episodic), len(result.semantic),
            )

            context = (
                result.render(
                    max_episodic=MAX_EPISODIC_IN_PROMPT,
                    max_semantic=MAX_SEMANTIC_IN_PROMPT,
                )
                + "\n\n"
                + _build_history_section(request.history)
                + "\n"
                + request.message
            )
            system_prompt = (
                f"{MEMORY_ANSWER_PROMPT}\n\nUser ID: {request.user_id}\n\n{context}"
            )

            # Phase 2a: consumer-owned answer generation
            async for chunk in chat_llm.stream(
                system_prompt, request.message, call_label="answer"
            ):
                accumulated += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            yield (
                "data: "
                + json.dumps({"type": "done", "full_content": accumulated})
                + "\n\n"
            )

            # Phase 2b: close the loop + manage, fire-and-forget
            asyncio.create_task(
                _post_answer_tasks(memory, request, result, accumulated)
            )
        except Exception as e:
            logger.error("Chat stream error for user %s: %s", request.user_id, e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _post_answer_tasks(
    memory: Memory, request: ChatRequest, result, answer_text: str
) -> None:
    """report_usage (reconsolidation) + manage, after the stream completes."""
    try:
        report = await memory.report_usage_async(result, answer_text)
        logger.info(
            "reconsolidation for user %s: used=%d assigned=%d",
            request.user_id, len(report.used_memory_ids), len(report.assignments),
        )
    except Exception as e:  # noqa: BLE001 - background task isolation
        logger.error("report_usage failed for user %s: %s", request.user_id, e)
    try:
        await memory.manage_async(
            user_text=request.message,
            assistant_text=answer_text,
            user_id=request.user_id,
            chat_id=request.chat_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Background manage failed for user %s: %s", request.user_id, e)
