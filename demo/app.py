"""Gradio demo — a plain consumer of the neuramem public API.

Slimmed per implementation plan step 4: no private attribute access at
all (the legacy demo reached into memory._llm_client,
memory._memory_usage_judge, memory._store and even memory._store._client).
Answer generation uses the demo's own LLM instance; the closed loop goes
through report_usage_async; panels read the store port via the public
`memory.store` property.
"""

import asyncio
import logging
import os
import time

import gradio as gr
import dotenv

from neuramem.config import MemoryConfig, StoreConfig
from neuramem.core.models import MemoryFilter
from neuramem.llm.openai_adapter import OpenAILLM
from neuramem.memory import Memory
from neuramem.prompts import MEMORY_ANSWER_PROMPT

dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEMO_EPISODIC_IN_PROMPT = 3
DEMO_SEMANTIC_IN_PROMPT = 3
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

    def _context(self, rendered: str) -> str:
        pairs = self.chat_history[-DEMO_HISTORY_PAIRS * 2:]
        history_lines = [
            f"  {msg['role']}: {msg['content']}" for msg in pairs
        ] or ["  (no history yet)"]
        return (
            f"{rendered}\n\nHere are the history messages:\n"
            + "\n".join(history_lines)
            + "\n\nHere is the current user message:\n"
        )

    async def _answer_stream(self, message: str):
        result = await self.memory.search_async(message, self.current_user_id)
        context = self._context(
            result.render(
                max_episodic=DEMO_EPISODIC_IN_PROMPT,
                max_semantic=DEMO_SEMANTIC_IN_PROMPT,
            )
        )
        system_prompt = f"{MEMORY_ANSWER_PROMPT}\n\n{context}"
        async for chunk in self.answer_llm.stream(
            system_prompt, message, call_label="answer"
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
        try:
            if result is not None:
                report = await self.memory.report_usage_async(result, answer)
                logger.info(
                    "reconsolidation: used=%s assigned=%s",
                    report.used_memory_ids, report.assignments,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("report_usage failed: %s", e)
        try:
            await self.memory.manage_async(
                message, answer, self.current_user_id,
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
