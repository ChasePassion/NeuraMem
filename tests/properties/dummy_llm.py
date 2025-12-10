"""Deterministic LLM stub for property tests to avoid external API calls."""

import json


class DummyLLMClient:
    """Stub LLM client that returns predictable responses."""

    def chat(self, system_prompt: str, user_message: str) -> str:
        # Default empty response for reconsolidation tests
        return ""

    def chat_json(self, system_prompt: str, user_message: str, default=None):
        if default is None:
            default = {"add": [], "update": [], "delete": []}

        def _wrap_response(data):
            """Wrap response in the standard chat_json format."""
            return {
                "parsed_data": data,
                "raw_response": json.dumps(data),
                "model": "dummy-model",
                "success": True
            }

        try:
            data = json.loads(user_message)
        except Exception:
            return _wrap_response(default)

        # Special handling for EPISODIC_MEMORY_MANAGER format
        if isinstance(data, dict) and "current_turn" in data and "episodic_memories" in data:
            current_turn = data.get("current_turn", {})
            user_text = current_turn.get("user", "")
            assistant_text = current_turn.get("assistant", "")
            existing_memories = data.get("episodic_memories", [])
            
            # Simple CRUD decision logic for testing
            lower = user_text.lower()
            
            # Check for greetings and chitchat that should NOT be stored
            greeting_phrases = {
                "good morning", "good evening", "good night", "早安", "早上好", 
                "晚上好", "晚安", "在吗", "hi", "hello", "hey", "haha", "哈哈",
                "谢谢", "thanks", "ok", "好的", "👍", "😊"
            }
            if lower.strip() in greeting_phrases or (len(lower.strip()) <= 8 and any(g in lower for g in greeting_phrases)):
                return _wrap_response({"add": [], "update": [], "delete": []})
            
            # Check for knowledge questions that should NOT be stored
            question_keywords = ["?", "？", "what", "how", "why", "who", "where", "什么", "吗"]
            if any(q in lower for q in question_keywords) and not any(keyword in lower for keyword in ["记住", "remember", "我是", "我叫", "我住", "我喜欢"]):
                return _wrap_response({"add": [], "update": [], "delete": []})
            
            # Check for personal information that should be stored
            positive_keywords = [
                "remember", "记住", "学生", "专业", "研究", "项目", "engineer", 
                "advisor", "导师", "喜欢", "住", "habit", "hobby", "major", 
                "thesis", "university", "college", "phd", "我是", "我叫", "我住"
            ]
            
            should_store = any(k in lower for k in positive_keywords) or len(user_text.strip()) >= 8
            
            if should_store:
                # Check for updates to existing memories
                existing_texts = [mem.get("text", "") for mem in existing_memories]
                
                # Simple update logic: if user mentions change, update existing memory
                if any(change_word in lower for change_word in ["现在", "已经", "变成", "changed", "now", "currently"]):
                    for i, existing_text in enumerate(existing_texts):
                        if any(keyword in existing_text.lower() for keyword in ["学生", "student", "工程师", "engineer"]):
                            return _wrap_response({
                                "add": [],
                                "update": [{
                                    "id": existing_memories[i].get("id", 1),
                                    "old_text": existing_text,
                                    "new_text": user_text
                                }],
                                "delete": []
                            })
                
                # Check for deletions (rare case)
                if any(delete_word in lower for delete_word in ["删除", "delete", "忘记", "forget"]):
                    for i, existing_text in enumerate(existing_texts):
                        if any(keyword in existing_text.lower() for keyword in user_text.lower().split()):
                            return _wrap_response({
                                "add": [],
                                "update": [],
                                "delete": [{"id": existing_memories[i].get("id", 1)}]
                            })
                
                # Default: add new memory
                return _wrap_response({
                    "add": [{"text": user_text}],
                    "update": [],
                    "delete": []
                })
            
            return _wrap_response({"add": [], "update": [], "delete": []})

        # For other prompts, just return the provided default to satisfy invariants
        return _wrap_response(default)
