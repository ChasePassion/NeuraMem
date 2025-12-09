"""
叙事记忆分组问题诊断测试

这个测试会逐步执行每个环节，并在每个环节输出详细的诊断信息，
帮助定位问题出在哪个步骤。

测试方法：
- 假设某个步骤成功（使用mock数据）
- 检查后续步骤是否正常工作
- 如果后续步骤正常，说明问题在被mock的步骤
"""

import pytest
import time
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class MockMemoryRecord:
    """模拟MemoryRecord"""
    id: int
    user_id: str
    memory_type: str
    ts: int
    chat_id: str
    text: str
    distance: float = 0.0


class NarrativeDiagnosis:
    """叙事记忆分组诊断"""
    
    def __init__(self):
        self.results = {}
    
    def diagnose_step1_search(self):
        """诊断步骤1-3：检索记忆"""
        print("\n" + "="*70)
        print("诊断步骤1-3：检索记忆")
        print("="*70)
        
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"diag_step1_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "diag_user"
        chat_id = "diag_chat"
        
        try:
            # 插入测试记忆
            test_text = "用户正在学习Python编程"
            embeddings = memory._embedding_client.encode([test_text])
            
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()),
                "chat_id": chat_id,
                "text": test_text,
                "vector": embeddings[0],
                "group_id": -1,
            }
            
            inserted_ids = memory._store.insert([entity])
            
            # 检索
            results = memory.search("Python", user_id)
            
            episodic = results.get("episodic", [])
            
            if episodic:
                print(f"✅ 检索成功，返回 {len(episodic)} 条记忆")
                for mem in episodic:
                    print(f"   ID={mem.id}, text='{mem.text}'")
                self.results["step1_search"] = "PASS"
            else:
                print("❌ 检索失败，没有返回记忆")
                self.results["step1_search"] = "FAIL"
            
            return episodic
            
        finally:
            memory._store.drop_collection()
    
    def diagnose_step6_judge(self):
        """诊断步骤6：MemoryUsageJudge"""
        print("\n" + "="*70)
        print("诊断步骤6：MemoryUsageJudge")
        print("="*70)
        
        from src.memory_system.processors.memory_usage_judge import MemoryUsageJudge
        from src.memory_system.clients import LLMClient
        import os
        
        llm_client = LLMClient(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            model="deepseek-chat"
        )
        
        judge = MemoryUsageJudge(llm_client)
        
        # 测试数据
        episodic_memories = ["用户正在学习Python编程"]
        
        context = f"""Here are the episodic memories:
1. {episodic_memories[0]}

Here are the task:
你记得我在学什么吗？"""
        
        final_reply = "是的，我记得你正在学习Python编程！"
        
        used_memories = judge.judge_used_memories(
            system_prompt=context,
            episodic_memories=episodic_memories,
            semantic_memories=[],
            message_history=[],
            final_reply=final_reply
        )
        
        print(f"输入的情景记忆: {episodic_memories}")
        print(f"模型回复: {final_reply}")
        print(f"judge返回: {used_memories}")
        
        if used_memories:
            print(f"✅ judge返回了 {len(used_memories)} 条使用的记忆")
            
            # 检查文本是否完全匹配
            exact_match = episodic_memories[0] in used_memories
            if exact_match:
                print("✅ 返回的文本与原始文本完全匹配")
                self.results["step6_judge"] = "PASS"
            else:
                print("⚠️ 返回的文本与原始文本不完全匹配！")
                print(f"   原始: '{episodic_memories[0]}'")
                print(f"   返回: '{used_memories[0] if used_memories else 'N/A'}'")
                self.results["step6_judge"] = "PARTIAL"
        else:
            print("❌ judge返回空列表")
            self.results["step6_judge"] = "FAIL"
        
        return used_memories, episodic_memories
    
    def diagnose_step6_7_matching(self):
        """诊断步骤6-7：文本到ID匹配"""
        print("\n" + "="*70)
        print("诊断步骤6-7：文本到ID匹配")
        print("="*70)
        
        # 模拟数据
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic",
                               ts=1000, chat_id="c1", text="用户正在学习Python编程"),
            ]
        }
        
        # 测试不同的匹配情况
        test_cases = [
            ("精确匹配", ["用户正在学习Python编程"]),
            ("末尾空格", ["用户正在学习Python编程 "]),
            ("末尾换行", ["用户正在学习Python编程\n"]),
            ("部分文本", ["学习Python编程"]),
            ("空列表", []),
        ]
        
        for case_name, used_texts in test_cases:
            used_ids = []
            for mem in relevant_memories.get("episodic", []):
                if mem.text in used_texts:
                    used_ids.append(mem.id)
            
            status = "✅" if used_ids else "❌"
            print(f"{status} {case_name}: used_texts={used_texts} -> matched_ids={used_ids}")
        
        self.results["step6_7_matching"] = "INFO"
    
    def diagnose_step7_assign(self):
        """诊断步骤7：assign_to_narrative_group"""
        print("\n" + "="*70)
        print("诊断步骤7：assign_to_narrative_group")
        print("="*70)
        
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"diag_step7_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "diag_user_step7"
        chat_id = "diag_chat"
        
        try:
            # 插入测试记忆
            test_text = "用户正在测试叙事组"
            embeddings = memory._embedding_client.encode([test_text])
            
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()),
                "chat_id": chat_id,
                "text": test_text,
                "vector": embeddings[0],
                "group_id": -1,
            }
            
            inserted_ids = memory._store.insert([entity])
            memory_id = inserted_ids[0]
            print(f"插入的记忆ID: {memory_id}")
            
            # 调用assign
            result = memory.assign_to_narrative_group([memory_id], user_id)
            print(f"assign返回: {result}")
            
            if memory_id in result:
                group_id = result[memory_id]
                print(f"✅ 记忆 {memory_id} 被分配到组 {group_id}")
                
                # 验证groups collection
                groups_collection = f"groups_{user_id}"
                if memory._store._client.has_collection(groups_collection):
                    groups = memory._store._client.query(
                        collection_name=groups_collection,
                        filter=f"user_id == '{user_id}'",
                        output_fields=["group_id", "size"]
                    )
                    print(f"groups collection内容: {groups}")
                    
                    if groups:
                        print("✅ groups collection有数据")
                        self.results["step7_assign"] = "PASS"
                    else:
                        print("❌ groups collection存在但没有数据")
                        self.results["step7_assign"] = "FAIL"
                else:
                    print(f"❌ groups collection不存在")
                    self.results["step7_assign"] = "FAIL"
            else:
                print(f"❌ 记忆 {memory_id} 没有被分配")
                self.results["step7_assign"] = "FAIL"
            
        finally:
            memory._store.drop_collection()
            groups_collection = f"groups_{user_id}"
            if memory._store._client.has_collection(groups_collection):
                memory._store._client.drop_collection(groups_collection)
    
    def diagnose_full_flow_with_mock_judge(self):
        """使用mock的judge测试完整流程
        
        如果这个测试通过，说明问题在judge步骤
        """
        print("\n" + "="*70)
        print("诊断：使用mock的judge测试完整流程")
        print("="*70)
        
        from src.memory_system import Memory, MemoryConfig
        
        config = MemoryConfig()
        config.collection_name = f"diag_mock_judge_{int(time.time())}"
        
        memory = Memory(config)
        user_id = "diag_user_mock"
        chat_id = "diag_chat"
        
        try:
            # 插入测试记忆
            test_text = "用户正在学习Python编程"
            embeddings = memory._embedding_client.encode([test_text])
            
            entity = {
                "user_id": user_id,
                "memory_type": "episodic",
                "ts": int(time.time()),
                "chat_id": chat_id,
                "text": test_text,
                "vector": embeddings[0],
                "group_id": -1,
            }
            
            inserted_ids = memory._store.insert([entity])
            memory_id = inserted_ids[0]
            print(f"插入的记忆ID: {memory_id}")
            
            # 检索
            results = memory.search("Python", user_id)
            episodic = results.get("episodic", [])
            print(f"检索到的记忆: {[(m.id, m.text) for m in episodic]}")
            
            # MOCK: 假设judge返回了正确的文本
            mock_used_texts = [test_text]  # 精确匹配原始文本
            print(f"MOCK judge返回: {mock_used_texts}")
            
            # 文本到ID匹配
            used_ids = []
            for mem in episodic:
                if mem.text in mock_used_texts:
                    used_ids.append(mem.id)
            print(f"匹配到的ID: {used_ids}")
            
            # 调用assign
            if used_ids:
                result = memory.assign_to_narrative_group(used_ids, user_id)
                print(f"assign返回: {result}")
                
                # 验证
                groups_collection = f"groups_{user_id}"
                if memory._store._client.has_collection(groups_collection):
                    groups = memory._store._client.query(
                        collection_name=groups_collection,
                        filter=f"user_id == '{user_id}'",
                        output_fields=["group_id", "size"]
                    )
                    
                    if groups:
                        print("✅ 使用mock judge后，叙事组创建成功！")
                        print("   这说明问题可能在真实的judge步骤")
                        self.results["mock_judge_flow"] = "PASS"
                    else:
                        print("❌ 即使使用mock judge，叙事组也没有创建")
                        self.results["mock_judge_flow"] = "FAIL"
                else:
                    print("❌ groups collection不存在")
                    self.results["mock_judge_flow"] = "FAIL"
            else:
                print("❌ 文本匹配失败")
                self.results["mock_judge_flow"] = "FAIL"
            
        finally:
            memory._store.drop_collection()
            groups_collection = f"groups_{user_id}"
            if memory._store._client.has_collection(groups_collection):
                memory._store._client.drop_collection(groups_collection)
    
    def run_all_diagnostics(self):
        """运行所有诊断"""
        print("\n" + "#"*70)
        print("# 叙事记忆分组问题诊断")
        print("#"*70)
        
        self.diagnose_step1_search()
        self.diagnose_step6_judge()
        self.diagnose_step6_7_matching()
        self.diagnose_step7_assign()
        self.diagnose_full_flow_with_mock_judge()
        
        print("\n" + "#"*70)
        print("# 诊断结果汇总")
        print("#"*70)
        
        for step, result in self.results.items():
            status = "✅" if result == "PASS" else ("⚠️" if result == "PARTIAL" else "❌")
            print(f"{status} {step}: {result}")
        
        print("\n" + "#"*70)
        print("# 问题定位建议")
        print("#"*70)
        
        if self.results.get("step6_judge") == "FAIL":
            print("❌ 问题可能在步骤6：MemoryUsageJudge返回空列表")
            print("   建议检查：")
            print("   - MEMORY_RELEVANCE_FILTER_PROMPT是否正确")
            print("   - LLM是否正确理解了判断逻辑")
        
        if self.results.get("step6_judge") == "PARTIAL":
            print("⚠️ 问题可能在步骤6-7之间：文本匹配不精确")
            print("   建议检查：")
            print("   - judge返回的文本是否与原始文本完全一致")
            print("   - 是否有空格、换行符等差异")
        
        if self.results.get("step7_assign") == "FAIL":
            print("❌ 问题可能在步骤7：assign_to_narrative_group执行失败")
            print("   建议检查：")
            print("   - Milvus连接是否正常")
            print("   - groups collection创建是否成功")
        
        if self.results.get("mock_judge_flow") == "PASS":
            print("\n💡 关键发现：使用mock judge后流程正常")
            print("   这强烈暗示问题在真实的judge步骤")


if __name__ == "__main__":
    diag = NarrativeDiagnosis()
    diag.run_all_diagnostics()
