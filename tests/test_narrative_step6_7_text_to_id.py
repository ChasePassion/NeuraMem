"""
测试步骤6-7之间的关键环节：文本到ID的匹配

这是最可能出问题的地方！
在demo/app.py的_process_memory_async中：
    for mem in relevant_memories.get("episodic", []):
        if mem.text in used_episodic_texts:
            used_memory_ids.append(mem.id)

问题可能出在：
1. judge返回的文本与原始文本不完全一致
2. 使用 "in" 操作符可能导致部分匹配问题
"""

import pytest
import time
from dataclasses import dataclass
from typing import List


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


class TestStep6To7TextToId:
    """测试文本到ID的匹配逻辑"""
    
    def test_text_matching_exact(self):
        """测试精确文本匹配"""
        # 模拟检索到的记忆
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic", 
                               ts=1000, chat_id="c1", text="用户正在学习Python编程"),
                MockMemoryRecord(id=2, user_id="u1", memory_type="episodic",
                               ts=1001, chat_id="c1", text="用户喜欢喝咖啡"),
            ]
        }
        
        # 模拟judge返回的使用记忆（精确匹配）
        used_episodic_texts = ["用户正在学习Python编程"]
        
        # 执行匹配逻辑（复制自demo/app.py）
        used_memory_ids = []
        for mem in relevant_memories.get("episodic", []):
            if mem.text in used_episodic_texts:
                used_memory_ids.append(mem.id)
        
        print(f"检索到的记忆: {[(m.id, m.text) for m in relevant_memories['episodic']]}")
        print(f"judge返回的使用文本: {used_episodic_texts}")
        print(f"匹配到的ID: {used_memory_ids}")
        
        assert 1 in used_memory_ids, "ID 1 应该被匹配到"
        assert 2 not in used_memory_ids, "ID 2 不应该被匹配到"
        
        print("✅ 精确匹配测试通过")
    
    def test_text_matching_with_whitespace_difference(self):
        """测试空白字符差异导致的匹配失败"""
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic",
                               ts=1000, chat_id="c1", text="用户正在学习Python编程"),
            ]
        }
        
        # 模拟judge返回的文本有额外空格
        used_episodic_texts = ["用户正在学习Python编程 "]  # 注意末尾空格
        
        used_memory_ids = []
        for mem in relevant_memories.get("episodic", []):
            if mem.text in used_episodic_texts:
                used_memory_ids.append(mem.id)
        
        print(f"原始文本: '{relevant_memories['episodic'][0].text}'")
        print(f"judge返回: '{used_episodic_texts[0]}'")
        print(f"匹配到的ID: {used_memory_ids}")
        
        if not used_memory_ids:
            print("❌ 空白字符差异导致匹配失败！这是一个潜在问题！")
        else:
            print("✅ 匹配成功")
    
    def test_text_matching_with_newline_difference(self):
        """测试换行符差异导致的匹配失败"""
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic",
                               ts=1000, chat_id="c1", text="用户正在学习Python编程"),
            ]
        }
        
        # 模拟judge返回的文本有换行符
        used_episodic_texts = ["用户正在学习Python编程\n"]
        
        used_memory_ids = []
        for mem in relevant_memories.get("episodic", []):
            if mem.text in used_episodic_texts:
                used_memory_ids.append(mem.id)
        
        print(f"原始文本: {repr(relevant_memories['episodic'][0].text)}")
        print(f"judge返回: {repr(used_episodic_texts[0])}")
        print(f"匹配到的ID: {used_memory_ids}")
        
        if not used_memory_ids:
            print("❌ 换行符差异导致匹配失败！这是一个潜在问题！")
        else:
            print("✅ 匹配成功")
    
    def test_text_matching_with_partial_text(self):
        """测试LLM返回部分文本导致的匹配失败"""
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic",
                               ts=1000, chat_id="c1", 
                               text="2024年12月9日，用户在北京学习Python编程，因为想成为程序员"),
            ]
        }
        
        # 模拟judge只返回了部分文本
        used_episodic_texts = ["用户在北京学习Python编程"]
        
        used_memory_ids = []
        for mem in relevant_memories.get("episodic", []):
            if mem.text in used_episodic_texts:
                used_memory_ids.append(mem.id)
        
        print(f"原始文本: '{relevant_memories['episodic'][0].text}'")
        print(f"judge返回: '{used_episodic_texts[0]}'")
        print(f"匹配到的ID: {used_memory_ids}")
        
        if not used_memory_ids:
            print("❌ 部分文本匹配失败！LLM可能没有返回完整的原始文本！")
        else:
            print("✅ 匹配成功")
    
    def test_text_matching_in_operator_direction(self):
        """测试 'in' 操作符的方向问题
        
        当前代码: if mem.text in used_episodic_texts
        这要求 mem.text 完全等于 used_episodic_texts 中的某个元素
        
        如果judge返回的是部分文本，这个匹配会失败
        """
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic",
                               ts=1000, chat_id="c1", text="用户正在学习Python编程"),
            ]
        }
        
        # 情况1：judge返回完整文本
        used_texts_1 = ["用户正在学习Python编程"]
        
        # 情况2：judge返回的文本包含原始文本（但不完全相等）
        used_texts_2 = ["我记得用户正在学习Python编程"]
        
        # 情况3：原始文本包含judge返回的文本
        used_texts_3 = ["学习Python"]
        
        for i, used_texts in enumerate([used_texts_1, used_texts_2, used_texts_3], 1):
            used_memory_ids = []
            for mem in relevant_memories.get("episodic", []):
                if mem.text in used_texts:
                    used_memory_ids.append(mem.id)
            
            print(f"\n情况{i}:")
            print(f"  原始文本: '{relevant_memories['episodic'][0].text}'")
            print(f"  judge返回: '{used_texts[0]}'")
            print(f"  匹配结果: {used_memory_ids}")
            
            if used_memory_ids:
                print(f"  ✅ 匹配成功")
            else:
                print(f"  ❌ 匹配失败")
    
    def test_empty_used_memories_list(self):
        """测试judge返回空列表的情况"""
        relevant_memories = {
            "episodic": [
                MockMemoryRecord(id=1, user_id="u1", memory_type="episodic",
                               ts=1000, chat_id="c1", text="用户正在学习Python编程"),
            ]
        }
        
        # judge返回空列表
        used_episodic_texts = []
        
        used_memory_ids = []
        for mem in relevant_memories.get("episodic", []):
            if mem.text in used_episodic_texts:
                used_memory_ids.append(mem.id)
        
        print(f"judge返回空列表")
        print(f"匹配到的ID: {used_memory_ids}")
        
        assert used_memory_ids == [], "空列表应该导致没有ID被匹配"
        print("✅ 空列表测试通过")


if __name__ == "__main__":
    test = TestStep6To7TextToId()
    print("="*60)
    print("测试1: 精确匹配")
    print("="*60)
    test.test_text_matching_exact()
    
    print("\n" + "="*60)
    print("测试2: 空白字符差异")
    print("="*60)
    test.test_text_matching_with_whitespace_difference()
    
    print("\n" + "="*60)
    print("测试3: 换行符差异")
    print("="*60)
    test.test_text_matching_with_newline_difference()
    
    print("\n" + "="*60)
    print("测试4: 部分文本")
    print("="*60)
    test.test_text_matching_with_partial_text()
    
    print("\n" + "="*60)
    print("测试5: in操作符方向")
    print("="*60)
    test.test_text_matching_in_operator_direction()
    
    print("\n" + "="*60)
    print("测试6: 空列表")
    print("="*60)
    test.test_empty_used_memories_list()
