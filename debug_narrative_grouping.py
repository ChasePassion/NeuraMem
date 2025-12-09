#!/usr/bin/env python3
"""
Debug script for narrative memory grouping functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory_system import Memory, MemoryConfig
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_narrative_grouping():
    """Debug the narrative grouping functionality."""
    
    print("🔍 Debugging Narrative Memory Grouping")
    print("=" * 50)
    
    # Initialize memory system
    user_id = "debug_user"
    config = MemoryConfig()
    config.collection_name = f"debug_memories_{user_id}"
    
    try:
        memory = Memory(config)
        print(f"✅ Memory system initialized for user: {user_id}")
    except Exception as e:
