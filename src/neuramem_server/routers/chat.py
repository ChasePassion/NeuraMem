"""SSE chat endpoint — the two-phase closed loop's server consumer.

Flow (architecture_target.md ch. 11), aligned with the benchmark runner
(same retrieval, same canonical answer prompt, same write-back):
1. search_async -> SearchResult (correlation token), ALL memories kept
2. server-owned LLM streams the answer (canonical build_answer_prompt,
   reference_date = current year, no 5/5 truncation)
3. on done, fire-and-forget: report_usage_async on the extracted final
   answer (closes the loop) + manage_async for the turn (product write
   path — the online counterpart of the benchmark's ingest phase)
"""

import asyncio
import datetime
import json
import logging
from typing import List

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from neuramem.core.ports import LLM
from neuramem.memory import Memory
from neuramem.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    extract_final_answer,
)
from neuramem_server.deps import get_chat_llm, get_memory_system
from neuramem_server.schemas import ChatMessage, ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# fire-and-forget tasks need a strong reference: asyncio keeps only weak
# refs, so unreferenced tasks can be garbage-collected mid-flight
_background_tasks: set[asyncio.Task] = set()

# conversation history is consumer-owned session state (not memory);
# recent turns enter the prompt for continuity
HISTORY_TURNS_IN_PROMPT = 6


def _history_block(history: List[ChatMessage]) -> str:
    """Render recent turns as context preceding the answer prompt."""
    if not history:
        return ""
    lines = [
        f"  {msg.role}: {msg.content}"
        for msg in history[-HISTORY_TURNS_IN_PROMPT:]
    ]
    return (
        "Here are the recent conversation messages for context:\n"
        + "\n".join(lines)
        + "\n\n"
    )


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

            # Phase 2a: consumer-owned answer generation — same builder
            # as the benchmark, all retrieved memories, current-year anchor
            user_prompt = _history_block(request.history) + build_answer_prompt(
                question=request.message,
                memories=result.episodic + result.semantic,
                reference_date=str(datetime.date.today().year),
            )
            async for chunk in chat_llm.stream(
                ANSWER_SYSTEM_PROMPT, user_prompt, call_label="answer"
            ):
                accumulated += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            yield (
                "data: "
                + json.dumps({"type": "done", "full_content": accumulated})
                + "\n\n"
            )

            # Phase 2b: close the loop + manage, fire-and-forget
            task = asyncio.create_task(
                _post_answer_tasks(memory, request, result, accumulated)
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
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
    """report_usage (reconsolidation) + manage, after the stream completes.

    The write-back sees the extracted final answer (reasoning stripped),
    exactly what the benchmark runner feeds report_usage_async.
    """
    final_answer = extract_final_answer(answer_text)
    try:
        report = await memory.report_usage_async(result, final_answer)
        if report.dropped_ids or report.malformed_count:
            logger.warning(
                "usage judge id anomalies for user %s: dropped=%s malformed=%d",
                request.user_id, report.dropped_ids, report.malformed_count,
            )
        logger.info(
            "reconsolidation for user %s: used=%d assigned=%d",
            request.user_id, len(report.used_memory_ids), len(report.assignments),
        )
    except Exception as e:  # noqa: BLE001 - background task isolation
        logger.error("report_usage failed for user %s: %s", request.user_id, e)
    try:
        await memory.manage_async(
            user_text=request.message,
            assistant_text=final_answer,
            user_id=request.user_id,
            chat_id=request.chat_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Background manage failed for user %s: %s", request.user_id, e)
