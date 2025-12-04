"""
Gradio-based visualization demo for AI Memory System.

Features:
- Left panel: Real-time memory display (episodic + semantic)
- Right panel: Chat interface
- Memory consolidation with progress display
- Scheduled consolidation support
"""

import gradio as gr
import asyncio
import time
import json
import threading
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_system import Memory, MemoryConfig, MemoryRecord, ConsolidationStats

# Setup logger
logger = logging.getLogger(__name__)


class MemoryDemoApp:
    """Main demo application class."""
    
    def __init__(self):
        """Initialize the demo application."""
        self.memory: Optional[Memory] = None
        self.current_user_id: str = "demo_user"
        self.chat_history: List[Tuple[str, str]] = []
        self.consolidation_log: List[str] = []
        self.is_consolidating: bool = False
        self.scheduled_task: Optional[threading.Timer] = None
        
    def initialize_memory_system(self, user_id: str) -> str:
        """Initialize or reinitialize the memory system."""
        try:
            if not user_id.strip():
                user_id = "demo_user"
            self.current_user_id = user_id.strip()
            
            config = MemoryConfig()
            config.collection_name = f"demo_memories_{self.current_user_id}"
            self.memory = Memory(config)
            self.chat_history = []
            self.consolidation_log = []
            
            return f"✅ 记忆系统初始化成功！用户ID: {self.current_user_id}"
        except Exception as e:
            return f"❌ 初始化失败: {str(e)}"
    
    def get_all_memories(self) -> str:
        """Get all memories for current user and format as display text."""
        if not self.memory:
            return "⚠️ 请先初始化记忆系统"
        
        try:
            # Query episodic memories
            episodic = self.memory._store.query(
                filter_expr=f'user_id == "{self.current_user_id}" and memory_type == "episodic"',
                output_fields=["id", "text", "hit_count", "ts", "metadata"],
                limit=100
            )
            
            # Query semantic memories
            semantic = self.memory._store.query(
                filter_expr=f'user_id == "{self.current_user_id}" and memory_type == "semantic"',
                output_fields=["id", "text", "hit_count", "ts", "metadata"],
                limit=100
            )
            
            output = []
            output.append(f"📊 记忆统计 - 用户: {self.current_user_id}")
            output.append(f"{'='*50}")
            output.append(f"情景记忆: {len(episodic)} 条 | 语义记忆: {len(semantic)} 条")
            output.append("")
            
            # Display episodic memories
            output.append("🎬 情景记忆 (Episodic)")
            output.append("-" * 40)
            if episodic:
                for mem in sorted(episodic, key=lambda x: x.get("ts", 0), reverse=True):
                    ts = mem.get("ts", 0)
                    time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "N/A"
                    hit = mem.get("hit_count", 0)
                    text = mem.get("text", "")[:80]
                    metadata = mem.get("metadata", {})
                    context = metadata.get("context", "")[:30]
                    output.append(f"[ID:{mem.get('id')}] 🕐{time_str} 💫{hit}次")
                    output.append(f"  📝 {text}...")
                    if context:
                        output.append(f"  📍 {context}")
                    output.append("")
            else:
                output.append("  (暂无情景记忆)")
                output.append("")
            
            # Display semantic memories
            output.append("🧠 语义记忆 (Semantic)")
            output.append("-" * 40)
            if semantic:
                for mem in sorted(semantic, key=lambda x: x.get("ts", 0), reverse=True):
                    ts = mem.get("ts", 0)
                    time_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "N/A"
                    hit = mem.get("hit_count", 0)
                    metadata = mem.get("metadata", {})
                    fact = metadata.get("fact", mem.get("text", ""))[:80]
                    output.append(f"[ID:{mem.get('id')}] 🕐{time_str} 💫{hit}次")
                    output.append(f"  💡 {fact}")
                    output.append("")
            else:
                output.append("  (暂无语义记忆)")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ 获取记忆失败: {str(e)}"

    async def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str]:
        """Process chat message with optimized flow: search → respond → async add memory."""
        if not self.memory:
            return "", history + [(message, "⚠️ 请先初始化记忆系统")], await asyncio.to_thread(self.get_all_memories)
        
        if not message.strip():
            return "", history, await asyncio.to_thread(self.get_all_memories)
        
        try:
            # 1. 准备消息和上下文
            prepared_messages = self._prepare_messages(message, history)
            
            # 2. 检索相关记忆（禁用同步重巩固，避免阻塞）
            relevant_memories = await asyncio.to_thread(
                self.memory.search,
                message,
                self.current_user_id,
                5,
                False  # reconsolidate off here; we handle asynchronously later
            )
            
            # 3. 构建完整上下文（传入 history）
            full_context = self._build_context_with_memories(message, relevant_memories, history)
            
            # 4. 调用LLM生成回复（放在线程池中执行）
            ai_response = await asyncio.to_thread(self._generate_response, full_context, prepared_messages)
            
            # 5. 异步巩固与写入：不阻塞当前回复
            asyncio.create_task(self._reconsolidate_async(message))
            asyncio.create_task(self._add_to_memory_async(message, history))
            
            # 构建最终响应
            final_response = ai_response
            new_history = history + [(message, final_response)]
            return "", new_history, await asyncio.to_thread(self.get_all_memories)
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            return "", history + [(message, error_msg)], await asyncio.to_thread(self.get_all_memories)
    
    def _prepare_messages(self, message: str, history: List[Tuple[str, str]]) -> List[Dict]:
        """准备和标准化消息，包含历史对话上下文。"""
        messages = []
        
        # 添加历史对话（最近50轮）
        for user_msg, ai_msg in history[-50:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        return messages
    
    def _fetch_relevant_memories(self, query: str) -> List[MemoryRecord]:
        """检索相关记忆。"""
        try:
            results = self.memory.search(
                query=query,
                user_id=self.current_user_id,
                limit=5,
                reconsolidate=True
            )
            return results
        except Exception as e:
            logger.warning(f"Failed to fetch memories: {e}")
            return []
    
    def _build_context_with_memories(self, message: str, memories: List[MemoryRecord], history: List[Tuple[str, str]]) -> str:
        """构建包含记忆的完整上下文。"""
        context_parts = []
        
        # 分离情景记忆和语义记忆
        episodic_memories = [mem for mem in memories if mem.memory_type == "episodic"]
        semantic_memories = [mem for mem in memories if mem.memory_type == "semantic"]
        
        # 1. 情景记忆部分
        context_parts.append("Here are the episodic memories:")
        if episodic_memories:
            for i, mem in enumerate(episodic_memories[:3], 1):
                context_parts.append(f"{i}. {mem.text}")
        else:
            context_parts.append("(No episodic memories)")
        context_parts.append("")
        
        # 2. 语义记忆部分
        context_parts.append("Here are the semantic memories:")
        if semantic_memories:
            for i, mem in enumerate(semantic_memories[:3], 1):
                context_parts.append(f"{i}. {mem.text}")
        else:
            context_parts.append("(No semantic memories)")
        context_parts.append("")
        
        # 3. 历史对话部分
        context_parts.append("Here are the history messages:")
        if history:
            for i, (user_msg, ai_msg) in enumerate(history[-3:], 1):
                context_parts.append(f"Turn {i}:")
                context_parts.append(f"  User: {user_msg}")
                context_parts.append(f"  Assistant: {ai_msg}")
        else:
            context_parts.append("(No history messages)")
        context_parts.append("")
        
        # 4. 当前任务
        context_parts.append("Here are the task:")
        context_parts.append(message)
        
        return "\n".join(context_parts)
    
    def _generate_response(self, context: str, messages: List[Dict]) -> str:
        """使用LLM生成回复。"""
        # 导入 MEMORY_ANSWER_PROMPT
        try:
            from prompts import MEMORY_ANSWER_PROMPT
            system_prompt = f"{MEMORY_ANSWER_PROMPT}\n\nUser ID: {self.current_user_id}\n\n{context}"
        except ImportError:
            system_prompt = f"""You are an AI assistant with long-term memory capabilities. User ID: {self.current_user_id}
Please answer based on the user's messages and relevant memories. If there are relevant memories, reflect that you remember the user's information in your response.
Maintain a friendly and natural conversation style.

{context}"""
        
        try:
            # 获取最后一条用户消息
            user_message = messages[-1]["content"] if messages else ""
            ai_response = self.memory._llm_client.chat(system_prompt, user_message)
            return ai_response
        except Exception as llm_error:
            return f"抱歉，我暂时无法生成回复。错误: {str(llm_error)}"
    
    async def _add_to_memory_async(self, message: str, history: List[Tuple[str, str]]) -> None:
        """异步添加记忆到后台（不阻塞 Gradio 事件循环）。"""
        try:
            chat_id = f"chat_{int(time.time())}"
            
            # 构建完整的对话上下文用于记忆提取
            conversation_context = self._build_conversation_context(message, history)
            
            # 优先使用异步版本，未实现时回退到线程池封装的同步接口
            if hasattr(self.memory, "add_async"):
                await self.memory.add_async(
                    text=conversation_context,
                    user_id=self.current_user_id,
                    chat_id=chat_id
                )
            else:
                await asyncio.to_thread(
                    self.memory.add,
                    conversation_context,
                    self.current_user_id,
                    chat_id
                )
        except Exception as e:
            logger.warning(f"Async memory add failed: {e}")

    async def _reconsolidate_async(self, query: str) -> None:
        """异步巩固检索到的情景记忆，避免阻塞响应。"""
        try:
            if hasattr(self.memory, "reconsolidate_async"):
                await self.memory.reconsolidate_async(query, self.current_user_id)
            else:
                # 回落：在后台线程调用带 reconsolidate 的 search
                await asyncio.to_thread(
                    self.memory.search,
                    query,
                    self.current_user_id,
                    5,
                    True
                )
        except Exception as e:
            logger.warning(f"Async reconsolidation failed: {e}")
    
    def _build_conversation_context(self, message: str, history: List[Tuple[str, str]]) -> str:
        """构建用于记忆提取的对话上下文。"""
        # 包含最近的对话历史（最多3轮）
        context_parts = []
        
        for user_msg, ai_msg in history[-3:]:
            context_parts.append(f"用户: {user_msg}")
            context_parts.append(f"助手: {ai_msg}")
        
        # 添加当前消息
        context_parts.append(f"用户: {message}")
        
        return "\n".join(context_parts)
    
    async def run_consolidation(self, progress=gr.Progress()) -> Tuple[str, str]:
        """Run memory consolidation with progress updates."""
        if not self.memory:
            return "⚠️ 请先初始化记忆系统", await asyncio.to_thread(self.get_all_memories)
        
        if self.is_consolidating:
            return "⏳ 巩固任务正在进行中...", await asyncio.to_thread(self.get_all_memories)
        
        self.is_consolidating = True
        self.consolidation_log = []
        
        try:
            self.consolidation_log.append(f"🚀 开始巩固 - {datetime.now().strftime('%H:%M:%S')}")
            progress(0.1, desc="正在查询记忆...")
            
            # Run consolidation in a worker thread to keep UI responsive
            stats = await asyncio.to_thread(self.memory.consolidate, user_id=self.current_user_id)
            
            progress(0.9, desc="巩固完成")
            
            # Build result log
            log = []
            log.append(f"✅ 巩固完成 - {datetime.now().strftime('%H:%M:%S')}")
            log.append(f"📊 处理统计:")
            log.append(f"  - 处理记忆数: {stats.memories_processed}")
            log.append(f"  - 创建语义数: {stats.semantic_created}")
            
            self.consolidation_log.extend(log)
            progress(1.0, desc="完成")
            
            return "\n".join(self.consolidation_log), await asyncio.to_thread(self.get_all_memories)
            
        except Exception as e:
            error = f"❌ 巩固失败: {str(e)}"
            self.consolidation_log.append(error)
            return "\n".join(self.consolidation_log), await asyncio.to_thread(self.get_all_memories)
        finally:
            self.is_consolidating = False
    
    async def reset_memories(self) -> Tuple[str, str]:
        """Reset all memories for current user."""
        if not self.memory:
            return "⚠️ 请先初始化记忆系统", ""
        
        try:
            count = await asyncio.to_thread(self.memory.reset, self.current_user_id)
            self.chat_history = []
            return f"✅ 已删除 {count} 条记忆", await asyncio.to_thread(self.get_all_memories)
        except Exception as e:
            return f"❌ 重置失败: {str(e)}", await asyncio.to_thread(self.get_all_memories)


def create_demo_interface():
    """Create and return the Gradio interface."""
    app = MemoryDemoApp()
    
    with gr.Blocks(title="AI Memory System Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 AI Memory System 可视化测试")
        gr.Markdown("基于认知心理学的AI长期记忆系统演示")
        
        with gr.Row():
            user_id_input = gr.Textbox(label="用户ID", value="demo_user", scale=2)
            init_btn = gr.Button("🔄 初始化系统", variant="primary", scale=1)
            init_status = gr.Textbox(label="状态", interactive=False, scale=2)
        
        with gr.Row():
            # Left panel - Memory display
            with gr.Column(scale=1):
                gr.Markdown("## 📚 记忆库")
                memory_display = gr.Textbox(
                    label="实时记忆状态",
                    lines=25,
                    max_lines=30,
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 刷新记忆", variant="secondary")
                
                gr.Markdown("### ⚙️ 记忆巩固")
                consolidate_btn = gr.Button("🔧 运行巩固", variant="primary")
                consolidation_output = gr.Textbox(label="巩固日志", lines=8, interactive=False)
                
                reset_btn = gr.Button("🗑️ 清空记忆", variant="stop")
                reset_output = gr.Textbox(label="操作结果", lines=2, interactive=False)
            
            # Right panel - Chat interface
            with gr.Column(scale=1):
                gr.Markdown("## 💬 对话测试")
                chatbot = gr.Chatbot(label="对话历史", height=400)
                msg_input = gr.Textbox(label="输入消息", placeholder="输入要记忆的内容...")
                send_btn = gr.Button("发送", variant="primary")
                
                gr.Markdown("### 💡 测试建议")
                gr.Markdown("""
                - 输入个人信息: "我是北京大学计算机专业的学生"
                - 明确记忆请求: "请记住我喜欢喝咖啡"
                - 项目信息: "我正在开发一个AI记忆系统"
                - 闲聊测试: "你好" (不会被记录)
                """)
        
        # Event handlers
        init_btn.click(
            fn=app.initialize_memory_system,
            inputs=[user_id_input],
            outputs=[init_status]
        ).then(
            fn=app.get_all_memories,
            outputs=[memory_display]
        )
        
        refresh_btn.click(fn=app.get_all_memories, outputs=[memory_display])
        
        send_btn.click(
            fn=app.chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, memory_display]
        )
        
        msg_input.submit(
            fn=app.chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, memory_display]
        )
        
        consolidate_btn.click(
            fn=app.run_consolidation,
            outputs=[consolidation_output, memory_display]
        )
        
        reset_btn.click(
            fn=app.reset_memories,
            outputs=[reset_output, memory_display]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_demo_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
