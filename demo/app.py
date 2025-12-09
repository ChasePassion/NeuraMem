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
from langfuse import observe, get_client
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
        self.chat_history: List[Dict[str, str]] = []
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
                output_fields=["id", "text", "ts", "group_id"],
                limit=100
            )
            
            # Query semantic memories
            semantic = self.memory._store.query(
                filter_expr=f'user_id == "{self.current_user_id}" and memory_type == "semantic"',
                output_fields=["id", "text", "ts"],
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
                    text = mem.get("text", "")
                    group_id = mem.get("group_id", -1)
                    group_info = f" [组:{group_id}]" if group_id != -1 else " [未分组]"
                    output.append(f"[ID:{mem.get('id')}] 时间:{time_str}{group_info}")
                    output.append(f"  内容: {text}")
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
                    fact = mem.get("text", "")
                    output.append(f"[ID:{mem.get('id')}] 时间:{time_str}")
                    output.append(f"  内容: {fact}")
                    output.append("")
            else:
                output.append("  (暂无语义记忆)")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ 获取记忆失败: {str(e)}"

    def get_narrative_groups(self) -> str:
        """Get all narrative groups for current user and format as display text."""
        if not self.memory:
            return "⚠️ 请先初始化记忆系统"
        
        try:
            # Get groups collection name for current user
            groups_collection_name = f"groups_{self.current_user_id}"
            
            # Check if groups collection exists
            if not self.memory._store._client.has_collection(groups_collection_name):
                return f"📋 叙事组 - 用户: {self.current_user_id}\n\n(暂无叙事组)"
            
            # Query all groups for the user
            groups = self.memory._store._client.query(
                collection_name=groups_collection_name,
                filter=f'user_id == "{self.current_user_id}"',
                output_fields=["group_id", "size", "centroid_vector"],
                limit=1000
            )
            
            output = []
            output.append(f"📋 叙事组 - 用户: {self.current_user_id}")
            output.append(f"{'='*50}")
            output.append(f"叙事组总数: {len(groups)} 个")
            output.append("")
            
            if groups:
                # Sort groups by size (largest first)
                groups_sorted = sorted(groups, key=lambda x: x.get("size", 0), reverse=True)
                
                for group in groups_sorted:
                    group_id = group.get("group_id", 0)
                    size = group.get("size", 0)
                    
                    output.append(f"🔗 叙事组 [ID:{group_id}]")
                    output.append(f"   成员数量: {size}")
                    
                    # Get members of this group
                    try:
                        members = self.memory._store.query(
                            filter_expr=f'group_id == {group_id} and user_id == "{self.current_user_id}"',
                            output_fields=["id", "text", "ts"],
                            limit=1000
                        )
                        
                        if members:
                            output.append(f"   成员列表:")
                            for mem in sorted(members, key=lambda x: x.get("ts", 0), reverse=True):
                                ts = mem.get("ts", 0)
                                time_str = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "N/A"
                                text = mem.get("text", "")
                                # Truncate long text
                                if len(text) > 50:
                                    text = text[:47] + "..."
                                output.append(f"     [ID:{mem.get('id')}] {time_str} - {text}")
                        
                    except Exception as e:
                        output.append(f"   (获取成员失败: {str(e)})")
                    
                    output.append("")
            else:
                output.append("(暂无叙事组)")
                output.append("")
            
            # Add statistics
            try:
                # Count ungrouped episodic memories
                ungrouped = self.memory._store.query(
                    filter_expr=f'user_id == "{self.current_user_id}" and memory_type == "episodic" and group_id == -1',
                    output_fields=["id"],
                    limit=10000
                )
                
                total_episodic = self.memory._store.query(
                    filter_expr=f'user_id == "{self.current_user_id}" and memory_type == "episodic"',
                    output_fields=["id"],
                    limit=10000
                )
                
                output.append("📈 统计信息:")
                output.append(f"   总情景记忆: {len(total_episodic)} 条")
                output.append(f"   已分组记忆: {len(total_episodic) - len(ungrouped)} 条")
                output.append(f"   未分组记忆: {len(ungrouped)} 条")
                if len(total_episodic) > 0:
                    grouped_ratio = (len(total_episodic) - len(ungrouped)) / len(total_episodic) * 100
                    output.append(f"   分组比例: {grouped_ratio:.1f}%")
                
            except Exception as e:
                output.append(f"   (统计信息获取失败: {str(e)})")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ 获取叙事组失败: {str(e)}"

    @observe(as_type="agent") 
    async def chat(self, message: str, history: List[Any]) -> Tuple[str, List[Dict[str, str]], str]:
        """Process chat message with intelligent reconsolidation: search → respond → judge usage → reconsolidate used memories."""
        history_messages = self._normalize_history(history)
    
        get_client().update_current_trace(
            session_id=f"demo_chat_{self.current_user_id}_{int(time.time())}",
            user_id=self.current_user_id,
            tags=["demo_chat", "memory_system"],
            metadata={
                "app": "MemoryDemoApp",
                "message_length": len(message),
                "history_length": len(history_messages)
            }
        )
        
        if not self.memory:
            return "", history_messages + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "⚠️ 请先初始化记忆系统"}
            ], await asyncio.to_thread(self.get_all_memories)
        
        if not message.strip():
            return "", history_messages, await asyncio.to_thread(self.get_all_memories)
        
        try:
            # 1. 准备消息和上下文
            prepared_messages = self._prepare_messages(message, history_messages)
            
            # 2. 检索相关记忆
            relevant_memories = await asyncio.to_thread(
                self.memory.search,
                message,
                self.current_user_id
            )
            
            # 3. 构建完整上下文（传入 history）
            full_context = self._build_context_with_memories(message, relevant_memories, history_messages)
            
            # 4. 调用LLM生成回复（放在线程池中执行）
            ai_response = await asyncio.to_thread(self._generate_response, full_context, prepared_messages)
            
            # 5. 记忆管理
            asyncio.create_task(self._manage_memory_async(message, ai_response, history_messages))
            
            # 构建最终响应
            final_response = ai_response
            new_history = history_messages + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": final_response}
            ]
            return "", new_history, await asyncio.to_thread(self.get_all_memories)
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            return "", history_messages + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ], await asyncio.to_thread(self.get_all_memories)
    
    async def chat_stream(self, message: str, history: List[Any]):
        """Process chat message with streaming response and intelligent reconsolidation."""
        history_messages = self._normalize_history(history)
        
        if not self.memory:
            error_response = "⚠️ 请先初始化记忆系统"
            new_history = history_messages + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_response}
            ]
            yield new_history, await asyncio.to_thread(self.get_all_memories), await asyncio.to_thread(self.get_narrative_groups)
            return
        
        if not message.strip():
            yield history_messages, await asyncio.to_thread(self.get_all_memories), await asyncio.to_thread(self.get_narrative_groups)
            return
        
        try:
            # 1. 准备消息和上下文
            prepared_messages = self._prepare_messages(message, history_messages)
            
            # 2. 检索相关记忆
            relevant_memories = await asyncio.to_thread(
                self.memory.search,
                message,
                self.current_user_id
            )
            
            # 3. 构建完整上下文（传入 history）
            full_context = self._build_context_with_memories(message, relevant_memories, history_messages)
            
            # 4. 创建用于收集完整回复的队列
            response_queue = asyncio.Queue()
            
            # 5. 启动流式响应生成
            accumulated_response = ""
            new_history = history_messages + [{"role": "user", "content": message}]
            
            # 先添加用户消息到历史
            yield new_history, await asyncio.to_thread(self.get_all_memories), await asyncio.to_thread(self.get_narrative_groups)
            
            # 流式生成回复
            async for chunk in self._generate_response_stream(full_context, prepared_messages):
                accumulated_response += chunk
                current_history = new_history + [{"role": "assistant", "content": accumulated_response}]
                yield current_history, await asyncio.to_thread(self.get_all_memories), await asyncio.to_thread(self.get_narrative_groups)
            
            # 6. 将完整回复放入队列供记忆处理使用
            await response_queue.put(accumulated_response)
            
            # 6. 启动记忆处理任务（在后台异步执行）
            asyncio.create_task(self._process_memory_async(
                user_message=message,
                response_queue=response_queue,
                history_messages=history_messages,
                relevant_memories=relevant_memories,
                full_context=full_context
            ))
            
        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}"
            error_history = history_messages + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            yield error_history, await asyncio.to_thread(self.get_all_memories), await asyncio.to_thread(self.get_narrative_groups)
    
    def _normalize_history(self, history: List[Any]) -> List[Dict[str, str]]:
        """Normalize Chatbot history to the messages format Gradio expects."""
        normalized: List[Dict[str, str]] = []
        
        for item in history or []:
            if isinstance(item, dict) and "role" in item and "content" in item:
                normalized.append({"role": str(item["role"]), "content": str(item["content"])})
            elif hasattr(item, "role") and hasattr(item, "content"):
                role = getattr(item, "role", None)
                content = getattr(item, "content", None)
                if role is not None and content is not None:
                    normalized.append({"role": str(role), "content": str(content)})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, ai_msg = item
                normalized.append({"role": "user", "content": str(user_msg)})
                normalized.append({"role": "assistant", "content": str(ai_msg)})
        
        return normalized
    
    def _history_pairs(self, history: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Convert message-style history into user/assistant pairs for logging or prompts."""
        pairs: List[Tuple[str, str]] = []
        last_user: Optional[str] = None
        
        for msg in history:
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
            elif msg.get("role") == "assistant" and last_user is not None:
                pairs.append((last_user, msg.get("content", "")))
                last_user = None
        
        return pairs
    
    def _prepare_messages(self, message: str, history: List[Dict[str, str]]) -> List[Dict]:
        """准备和标准化消息，包含历史对话上下文。"""
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history[-50:]
            if isinstance(msg, dict) and "role" in msg and "content" in msg
        ]
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        return messages
    
    
    def _build_context_with_memories(self, message: str, memories: Dict[str, List[MemoryRecord]], history: List[Dict[str, str]]) -> str:
        """构建包含记忆的完整上下文。"""
        context_parts = []
        history_pairs = self._history_pairs(history)
        
        # 1. 情景记忆部分
        context_parts.append("Here are the episodic memories:")
        episodic_memories = memories.get("episodic", [])
        if episodic_memories:
            for i, mem in enumerate(episodic_memories[:3], 1):
                context_parts.append(f"{i}. {mem.text}")
        else:
            context_parts.append("(No episodic memories)")
        context_parts.append("")
        
        # 2. 语义记忆部分
        context_parts.append("Here are the semantic memories:")
        semantic_memories = memories.get("semantic", [])
        if semantic_memories:
            for i, mem in enumerate(semantic_memories[:3], 1):
                context_parts.append(f"{i}. {mem.text}")
        else:
            context_parts.append("(No semantic memories)")
        context_parts.append("")
        
        # 3. 历史对话部分
        context_parts.append("Here are the history messages:")
        if history_pairs:
            for i, (user_msg, ai_msg) in enumerate(history_pairs[-3:], 1):
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
    
    async def _generate_response_stream(self, context: str, messages: List[Dict]):
        """使用LLM流式生成回复。"""
        # 导入 MEMORY_ANSWER_PROMPT
        try:
            from prompts import MEMORY_ANSWER_PROMPT
            system_prompt = f"{MEMORY_ANSWER_PROMPT}\n\n{context}"
        except ImportError:
            system_prompt = f"""You are an AI assistant with long-term memory capabilities. User ID: {self.current_user_id}
Please answer based on the user's messages and relevant memories. If there are relevant memories, reflect that you remember the user's information in your response.
Maintain a friendly and natural conversation style.

{context}"""
        
        try:
            # 获取最后一条用户消息
            user_message = messages[-1]["content"] if messages else ""
            
            # 在线程池中执行流式调用
            response_stream = await asyncio.to_thread(
                self.memory._llm_client.chat_stream, system_prompt, user_message
            )
            
            accumulated_response = ""
            for chunk in response_stream:
                accumulated_response += chunk
                yield chunk
                
        except Exception as llm_error:
            yield f"抱歉，我暂时无法生成回复。错误: {str(llm_error)}"
    
    async def _manage_memory_async(self, user_message: str, assistant_message: str, history: List[Dict[str, str]]) -> None:
        """异步管理记忆到后台（不阻塞 Gradio 事件循环）。"""
        try:
            chat_id = f"chat_{int(time.time())}"
            
            # 优先使用异步版本，未实现时回退到线程池封装的同步接口
            if hasattr(self.memory, "manage_async"):
                await self.memory.manage_async(
                    user_text=user_message,
                    assistant_text=assistant_message,
                    user_id=self.current_user_id,
                    chat_id=chat_id
                )
            else:
                await asyncio.to_thread(
                    self.memory.manage,
                    user_message,
                    assistant_message,
                    self.current_user_id,
                    chat_id
                )
        except Exception as e:
            logger.warning(f"Async memory manage failed: {e}")

    async def _manage_memory_async_with_queue(self, user_message: str, response_queue: asyncio.Queue, history: List[Dict[str, str]]) -> None:
        """异步管理记忆到后台（使用队列获取完整回复，确保在流式输出结束后调用）。"""
        try:
            # 从队列获取完整回复
            assistant_message = await response_queue.get()
            chat_id = f"chat_{int(time.time())}"
            
            # 优先使用异步版本，未实现时回退到线程池封装的同步接口
            if hasattr(self.memory, "manage_async"):
                await self.memory.manage_async(
                    user_text=user_message,
                    assistant_text=assistant_message,
                    user_id=self.current_user_id,
                    chat_id=chat_id
                )
            else:
                await asyncio.to_thread(
                    self.memory.manage,
                    user_message,
                    assistant_message,
                    self.current_user_id,
                    chat_id
                )
        except Exception as e:
            logger.warning(f"Async memory manage with queue failed: {e}")

    async def _process_memory_async(
        self,
        user_message: str,
        response_queue: asyncio.Queue,
        history_messages: List[Dict[str, str]],
        relevant_memories: Dict[str, List[MemoryRecord]],
        full_context: str
    ) -> None:
        """异步处理记忆：判断使用 → 叙事分组 → manage"""
        try:
            # 1. 从队列获取完整回复
            assistant_message = await response_queue.get()
            
            # 2. 调用 MemoryUsageJudge 判断哪些情景记忆被使用
            episodic_texts = [mem.text for mem in relevant_memories.get("episodic", [])]
            semantic_texts = [mem.text for mem in relevant_memories.get("semantic", [])]
            
            used_episodic_texts = await asyncio.to_thread(
                self.memory._memory_usage_judge.judge_used_memories,
                system_prompt=full_context,
                episodic_memories=episodic_texts,
                semantic_memories=semantic_texts,
                message_history=history_messages,
                final_reply=assistant_message
            )
            
            # 3. 找出被使用的情景记忆的 ID
            used_memory_ids = []
            for mem in relevant_memories.get("episodic", []):
                if mem.text in used_episodic_texts:
                    used_memory_ids.append(mem.id)
            
            # 4. 对被使用的情景记忆执行叙事分组
            if used_memory_ids:
                await asyncio.to_thread(
                    self.memory.assign_to_narrative_group,
                    memory_ids=used_memory_ids,
                    user_id=self.current_user_id
                )
                logger.info(f"Assigned {len(used_memory_ids)} episodic memories to narrative groups")
            
            # 5. 执行 manage 管理记忆
            chat_id = f"chat_{int(time.time())}"
            if hasattr(self.memory, "manage_async"):
                await self.memory.manage_async(
                    user_text=user_message,
                    assistant_text=assistant_message,
                    user_id=self.current_user_id,
                    chat_id=chat_id
                )
            else:
                await asyncio.to_thread(
                    self.memory.manage,
                    user_message,
                    assistant_message,
                    self.current_user_id,
                    chat_id
                )
        except Exception as e:
            logger.warning(f"Async memory processing failed: {e}")

    async def _add_to_memory_async(self, message: str, history: List[Dict[str, str]]) -> None:
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

    
    def _build_conversation_context(self, message: str, history: List[Dict[str, str]]) -> str:
        """构建用于记忆提取的对话上下文。"""
        # 包含最近的对话历史（最多3轮）
        context_parts = []
        history_pairs = self._history_pairs(history)
        
        for user_msg, ai_msg in history_pairs[-3:]:
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
                    lines=20,
                    max_lines=25,
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 刷新记忆", variant="secondary")
                
                gr.Markdown("### ⚙️ 记忆巩固")
                consolidate_btn = gr.Button("🔧 运行巩固", variant="primary")
                consolidation_output = gr.Textbox(label="巩固日志", lines=6, interactive=False)
                
                reset_btn = gr.Button("🗑️ 清空记忆", variant="stop")
                reset_output = gr.Textbox(label="操作结果", lines=2, interactive=False)
            
            # Middle panel - Chat interface
            with gr.Column(scale=1):
                gr.Markdown("## 💬 对话测试")
                chatbot = gr.Chatbot(label="对话历史", height=400, type='messages')
                msg_input = gr.Textbox(label="输入消息", placeholder="输入要记忆的内容...")
                send_btn = gr.Button("发送", variant="primary")
                
                gr.Markdown("### 💡 测试建议")
                gr.Markdown("""
                - 输入个人信息: "我是北京大学计算机专业的学生"
                - 明确记忆请求: "请记住我喜欢喝咖啡"
                - 项目信息: "我正在开发一个AI记忆系统"
                - 闲聊测试: "你好" (不会被记录)
                """)
            
            # Right panel - Narrative groups display
            with gr.Column(scale=1):
                gr.Markdown("## 📋 叙事组")
                groups_display = gr.Textbox(
                    label="叙事组状态",
                    lines=25,
                    max_lines=30,
                    interactive=False
                )
                refresh_groups_btn = gr.Button("🔄 刷新叙事组", variant="secondary")
                
                gr.Markdown("### 📊 叙事统计")
                gr.Markdown("""
                **叙事记忆功能说明:**
                - 🔗 叙事组将相关的情景记忆组织在一起
                - 📈 显示分组统计和成员信息
                - 🎯 只有被实际使用的记忆才会分组
                - 🔄 自动维护组中心向量
                """)
        
        # Event handlers
        init_btn.click(
            fn=app.initialize_memory_system,
            inputs=[user_id_input],
            outputs=[init_status]
        ).then(
            fn=app.get_all_memories,
            outputs=[memory_display]
        ).then(
            fn=app.get_narrative_groups,
            outputs=[groups_display]
        )
        
        refresh_btn.click(fn=app.get_all_memories, outputs=[memory_display])
        
        refresh_groups_btn.click(fn=app.get_narrative_groups, outputs=[groups_display])
        
        send_btn.click(
            fn=app.chat_stream,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, memory_display, groups_display]
        )
        
        msg_input.submit(
            fn=app.chat_stream,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, memory_display, groups_display]
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
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
