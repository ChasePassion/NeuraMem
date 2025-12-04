"""
Gradio-based visualization demo for AI Memory System.

Features:
- Left panel: Real-time memory display (episodic + semantic)
- Right panel: Chat interface
- Memory consolidation with progress display
- Scheduled consolidation support
"""

import gradio as gr
import time
import json
import threading
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_system import Memory, MemoryConfig, MemoryRecord, ConsolidationStats


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

    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str]:
        """Process chat message and update memories, then call DeepSeek for response."""
        if not self.memory:
            return "", history + [(message, "⚠️ 请先初始化记忆系统")], self.get_all_memories()
        
        if not message.strip():
            return "", history, self.get_all_memories()
        
        try:
            chat_id = f"chat_{int(time.time())}"
            
            # Add memory from user message
            ids = self.memory.add(
                text=message,
                user_id=self.current_user_id,
                chat_id=chat_id
            )
            
            # Search relevant memories
            results = self.memory.search(
                query=message,
                user_id=self.current_user_id,
                limit=5,
                reconsolidate=True
            )
            
            # Build memory context for LLM
            memory_context = ""
            if results:
                memory_context = "\n\n相关记忆:\n"
                for r in results[:3]:
                    memory_context += f"- {r.text} (类型:{r.memory_type})\n"
            
            # Generate response using DeepSeek
            system_prompt = f"""你是一个具有长期记忆能力的AI助手。用户ID: {self.current_user_id}
请根据用户的消息和相关记忆来回答。如果有相关记忆，请在回答中体现出你记住了用户的信息。
保持友好、自然的对话风格。
{memory_context if memory_context else "当前无相关记忆"}"""
            
            try:
                ai_response = self.memory._llm_client.chat(system_prompt, message)
            except Exception as llm_error:
                ai_response = f"抱歉，我暂时无法生成回复。错误: {str(llm_error)}"
            
            # Add memory status info
            if ids:
                memory_status = f"\n\n💾 已记录此次对话"
            else:
                memory_status = "\n\n💭 此次对话未触发记忆存储"
            
            final_response = ai_response + memory_status
            new_history = history + [(message, final_response)]
            return "", new_history, self.get_all_memories()
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            return "", history + [(message, error_msg)], self.get_all_memories()
    
    def run_consolidation(self, progress=gr.Progress()) -> Tuple[str, str]:
        """Run memory consolidation with progress updates."""
        if not self.memory:
            return "⚠️ 请先初始化记忆系统", self.get_all_memories()
        
        if self.is_consolidating:
            return "⏳ 巩固任务正在进行中...", self.get_all_memories()
        
        self.is_consolidating = True
        self.consolidation_log = []
        
        try:
            self.consolidation_log.append(f"🚀 开始巩固 - {datetime.now().strftime('%H:%M:%S')}")
            progress(0.1, desc="正在查询记忆...")
            
            # Run consolidation
            stats = self.memory.consolidate(user_id=self.current_user_id)
            
            progress(0.9, desc="巩固完成")
            
            # Build result log
            log = []
            log.append(f"✅ 巩固完成 - {datetime.now().strftime('%H:%M:%S')}")
            log.append(f"📊 处理统计:")
            log.append(f"  - 处理记忆数: {stats.memories_processed}")
            log.append(f"  - 创建语义数: {stats.semantic_created}")
            
            self.consolidation_log.extend(log)
            progress(1.0, desc="完成")
            
            return "\n".join(self.consolidation_log), self.get_all_memories()
            
        except Exception as e:
            error = f"❌ 巩固失败: {str(e)}"
            self.consolidation_log.append(error)
            return "\n".join(self.consolidation_log), self.get_all_memories()
        finally:
            self.is_consolidating = False
    
    def reset_memories(self) -> Tuple[str, str]:
        """Reset all memories for current user."""
        if not self.memory:
            return "⚠️ 请先初始化记忆系统", ""
        
        try:
            count = self.memory.reset(self.current_user_id)
            self.chat_history = []
            return f"✅ 已删除 {count} 条记忆", self.get_all_memories()
        except Exception as e:
            return f"❌ 重置失败: {str(e)}", self.get_all_memories()


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
