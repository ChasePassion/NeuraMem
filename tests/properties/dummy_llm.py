"""Deterministic LLM stub for property tests to avoid external API calls."""

import json


class DummyLLMClient:
    """Stub LLM client that returns predictable responses."""

    def chat(self, system_prompt: str, user_message: str) -> str:
        # Default empty response for reconsolidation tests
        return ""

    def chat_json(self, system_prompt: str, user_message: str, default=None):
        if default is None:
            default = {}

        try:
            data = json.loads(user_message)
        except Exception:
            return default

        # Special handling for write-decider style inputs
        if isinstance(data, dict) and "turns" in data:
            turns = data.get("turns", [])
            contents = " ".join(str(t.get("content", "")) for t in turns)
            lower = contents.lower()

            greeting_phrases = {
                "good morning",
                "good evening",
                "good night",
                "早安",
                "早上好",
                "晚上好",
                "晚安",
                "在吗",
            }
            if lower.strip() in greeting_phrases:
                return {"write_episodic": False, "records": []}

            greetings = [
                "hi",
                "hello",
                "hey",
                "haha",
                "哈哈",
                "谢谢",
                "thanks",
                "ok",
                "好的",
                "👍",
                "😊",
            ]
            if len(contents.strip()) <= 8 and any(k in lower for k in greetings):
                return {"write_episodic": False, "records": []}

            positive_keywords = [
                "remember",
                "记住",
                "学生",
                "专业",
                "研究",
                "项目",
                "engineer",
                "advisor",
                "导师",
                "喜欢",
                "住",
                "habit",
                "hobby",
                "major",
                "thesis",
                "university",
                "college",
                "phd",
            ]
            question_keywords = ["?", "？", "what", "how", "why", "who", "where", "什么", "吗"]
            is_question = any(q in contents or lower.startswith(q) for q in question_keywords)
            should_store = (
                any(k in lower for k in positive_keywords)
                or (len(contents.strip()) >= 8 and not is_question)
            )

            if should_store:
                who = turns[-1].get("role", "user") if turns else "user"
                snippet = contents[:50] if contents else "detail"
                return {
                    "write_episodic": True,
                    "records": [
                        {
                            "who": who,
                            "text": contents or "memory",
                            "metadata": {
                                "context": snippet,
                                "thing": snippet,
                                "who": who,
                            },
                        }
                    ],
                }

            return {"write_episodic": False, "records": []}

        # For other prompts, just return the provided default to satisfy invariants
        return default
