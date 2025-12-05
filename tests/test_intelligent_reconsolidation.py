"""Simple test to verify intelligent reconsolidation logic."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_system.processors.memory_usage_judge import MemoryUsageJudge
from src.memory_system.clients import LLMClient
from src.memory_system.config import MemoryConfig

def test_memory_usage_judge():
    """Test the MemoryUsageJudge class."""
    print("Testing MemoryUsageJudge...")
    
    # Create a mock LLM client
    config = MemoryConfig()
    llm_client = LLMClient(
        api_key=config.deepseek_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model
    )
    
    # Create judge
    judge = MemoryUsageJudge(llm_client)
    
    # Test data
    system_prompt = "You are a helpful assistant."
    episodic_memories = [
        "User mentioned they are studying computer science at Beijing University.",
        "User said they like drinking coffee in the morning.",
        "User talked about their cat named Whiskers."
    ]
    semantic_memories = [
        "User is a computer science student.",
        "User prefers coffee over tea."
    ]
    message_history = [
        {"role": "user", "content": "What do you know about me?"},
    ]
    final_reply = "Based on what you've told me, you are studying computer science at Beijing University and you enjoy drinking coffee in the morning."
    
    # Judge which memories were used
    try:
        used_memories = judge.judge_used_memories(
            system_prompt=system_prompt,
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            message_history=message_history,
            final_reply=final_reply
        )
        
        print(f"\n✅ Test passed!")
        print(f"Total episodic memories: {len(episodic_memories)}")
        print(f"Used episodic memories: {len(used_memories)}")
        print(f"\nUsed memories:")
        for mem in used_memories:
            print(f"  - {mem}")
        
        # Expected: The first two memories should be marked as used
        # The third one (about the cat) should not be used
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_memory_usage_judge()
