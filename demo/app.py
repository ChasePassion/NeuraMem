"""Gradio demo — a plain consumer of the neuramem public API.

Slimmed per implementation plan step 4: no private attribute access at
all (the legacy demo reached into memory._llm_client,
memory._memory_usage_judge, memory._store and even memory._store._client).
Answer generation uses the demo's own LLM instance; the closed loop goes
through report_usage_async; panels read the store port via the public
`memory.store` property.
"""

import asyncio
import datetime
import logging
import os
import time

import gradio as gr
import dotenv

from neuramem.config import MemoryConfig, StoreConfig
from neuramem.core.models import MemoryFilter
from neuramem.llm.openai_adapter import OpenAILLM
from neuramem.memory import Memory
from neuramem.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
    extract_final_answer,
)

dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEMO_HISTORY_PAIRS = 3


class MemoryDemoApp:
    def __init__(self):
        self.current_user_id = "demo_user"
        self.chat_history: list[dict] = []
        self.memory: Memory | None = None
        self.answer_llm: OpenAILLM | None = None
        # strong refs for fire-and-forget tasks (asyncio holds weak refs)
        self._background_tasks: set[asyncio.Task] = set()

    # -- lifecycle ---------------------------------------------------------

    def switch_user(self, user_id: str):
        user_id = (user_id or "demo_user").strip() or "demo_user"
        self.current_user_id = user_id
        self.chat_history = []
        # per-user collections keep demo users isolated, legacy demo behavior
        config = MemoryConfig()
        store_kwargs = config.store.model_dump()
        store_kwargs["collection_name"] = f"demo_memories_{user_id}"
        store_kwargs["groups_collection_name"] = f"demo_groups_{user_id}"
        config.store = StoreConfig(_env_file=None, **store_kwargs)
        self.memory = Memory(config)
        self.answer_llm = OpenAILLM(config.llm)
        logger.info("demo initialized for user %s", user_id)
        return self.refresh_panels()

    # -- panels (store port, public property) --------------------------------

    async def _memory_list(self) -> str:
        records = await self.memory.store.query(
            MemoryFilter(user_id=self.current_user_id), limit=500
        )
        if not records:
            return "_(no memories yet)_"
        lines = [
            f"- `#{r.id}` [{r.memory_type}] {r.text}"
            + (f" _(group {r.group_id})_" if r.group_id != -1 else "")
            + (" _(retired)_" if r.retired else "")
            for r in records
        ]
        return "\n".join(lines)

    async def _groups_list(self) -> str:
        groups = await self.memory.store.list_groups(self.current_user_id)
        if not groups:
            return "_(no narrative groups yet — they form as you chat)_"
        lines = []
        for group in groups:
            members = await self.memory.store.get_group_members(
                group.group_id, self.current_user_id
            )
            lines.append(f"**Group {group.group_id}** ({group.size} members)")
            lines.extend(f"  - {m.text}" for m in members)
        return "\n".join(lines)

    def refresh_panels(self):
        memories, groups = asyncio.run(
            asyncio.gather(self._memory_list(), self._groups_list())
        )
        return memories, groups

    # -- chat ----------------------------------------------------------------

    def _history_block(self) -> str:
        pairs = self.chat_history[-DEMO_HISTORY_PAIRS * 2:]
        if not pairs:
            return ""
        history_lines = [f"  {msg['role']}: {msg['content']}" for msg in pairs]
        return (
            "Here are the recent conversation messages for context:\n"
            + "\n".join(history_lines)
            + "\n\n"
        )

    async def _answer_stream(self, message: str):
        # same chain as the benchmark runner: all retrieved memories into
        # the canonical answer prompt, current-year temporal anchor
        result = await self.memory.search_async(message, self.current_user_id)
        user_prompt = self._history_block() + build_answer_prompt(
            question=message,
            memories=result.episodic + result.semantic,
            reference_date=str(datetime.date.today().year),
        )
        async for chunk in self.answer_llm.stream(
            ANSWER_SYSTEM_PROMPT, user_prompt, call_label="answer"
        ):
            yield chunk, result

    async def chat_stream(self, message: str, history):
        accumulated = ""
        result = None
        try:
            async for chunk, search_result in self._answer_stream(message):
                accumulated += chunk
                result = search_result
                yield accumulated
        except Exception as e:  # noqa: BLE001 - surface to the UI
            yield f"(generation failed: {e})"
            return
        finally:
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": accumulated})

        # closed loop + manage, fire and forget
        task = asyncio.create_task(self._post_turn(message, accumulated, result))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _post_turn(self, message: str, answer: str, result):
        final_answer = extract_final_answer(answer)
        try:
            if result is not None:
                report = await self.memory.report_usage_async(result, final_answer)
                if report.dropped_ids or report.malformed_count:
                    logger.warning(
                        "usage judge id anomalies: dropped=%s malformed=%d",
                        report.dropped_ids, report.malformed_count,
                    )
                logger.info(
                    "reconsolidation: used=%s assigned=%s",
                    report.used_memory_ids, report.assignments,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("report_usage failed: %s", e)
        try:
            await self.memory.manage_async(
                message, final_answer, self.current_user_id,
                chat_id=f"chat_{int(time.time())}",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("manage failed: %s", e)

    # -- buttons ---------------------------------------------------------------

    def run_consolidation(self):
        stats = self.memory.consolidate(self.current_user_id)
        return (
            f"Processed {stats.memories_processed} episodic -> "
            f"{stats.semantic_created} semantic"
        ), *self.refresh_panels()

    def reset_memories(self):
        deleted = self.memory.reset(self.current_user_id)
        self.chat_history = []
        return f"Deleted {deleted} memories", *self.refresh_panels()


def create_demo_interface():
    app_ui = MemoryDemoApp()

    with gr.Blocks(title="NeuraMem Demo") as interface:
        gr.Markdown("# NeuraMem — memory demo (public API only)")
        with gr.Row():
            with gr.Column(scale=1):
                user_box = gr.Textbox(
                    label="User ID", value="demo_user"
                )
                switch_btn = gr.Button("Switch user / init")
                memory_panel = gr.Markdown("_(not initialized)_")
                consolidate_btn = gr.Button("Consolidate")
                consolidate_status = gr.Markdown()
                reset_btn = gr.Button("Reset memories")
            with gr.Column(scale=2):
                chat = gr.ChatInterface(
                    app_ui.chat_stream,
                    title=f"Chat with memory",
                )
            with gr.Column(scale=1):
                gr.Markdown("### Narrative groups")
                groups_panel = gr.Markdown("_(not initialized)_")

        def _after_switch():
            memories, groups = app_ui.refresh_panels()
            return memories, groups

        switch_btn.click(
            _after_switch, inputs=None, outputs=[memory_panel, groups_panel]
        )
        consolidate_btn.click(
            app_ui.run_consolidation,
            inputs=None,
            outputs=[consolidate_status, memory_panel, groups_panel],
        )
        reset_btn.click(
            app_ui.reset_memories,
            inputs=None,
            outputs=[consolidate_status, memory_panel, groups_panel],
        )
        user_box.submit(
            app_ui.switch_user,
            inputs=[user_box],
            outputs=[memory_panel, groups_panel],
        )
    return interface


if __name__ == "__main__":
    create_demo_interface().launch(
        server_name="0.0.0.0", server_port=int(os.getenv("DEMO_PORT", "7861"))
    )
