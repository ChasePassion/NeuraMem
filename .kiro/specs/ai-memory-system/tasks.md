# Implementation Plan

- [x] 1. Set up project structure and core configuration




  - [x] 1.1 Create project directory structure with src/, tests/, and config files


    - Create directories: `src/memory_system/`, `src/memory_system/processors/`, `tests/`, `tests/properties/`, `tests/integration/`
    - Create `__init__.py` files for all packages
    - Create `requirements.txt` with dependencies: pymilvus, openai, hypothesis, pytest
    - _Requirements: 1.1, 1.3_

  - [x] 1.2 Implement MemoryConfig dataclass

    - Define all configuration fields with defaults as specified in design
    - Include Milvus URI, OpenRouter credentials, model IDs, thresholds
    - _Requirements: 1.1, 1.3_
  - [x] 1.3 Write property test for dynamic field storage






    - **Property 1: Dynamic Field Storage Consistency**
    - **Validates: Requirements 1.4**

- [x] 2. Implement infrastructure layer clients







  - [x] 2.1 Implement EmbeddingClient class




    - Create OpenRouter API client for qwen/qwen3-embedding-4b model
    - Implement encode() method for batch text embedding
    - Implement dim property returning 2560
    - Add retry logic with exponential backoff (3 retries)
    - _Requirements: 1.3, 10.2_
  - [x] 2.2 Implement LLMClient class


    - Create OpenRouter API client for x-ai/grok-4.1-fast:free model
    - Implement chat() method for text responses
    - Implement chat_json() method with JSON parsing and safe fallback
    - Add retry logic with exponential backoff (3 retries)
    - _Requirements: 1.3, 10.2, 10.3_
  - [x] 2.3 Implement MilvusStore class


    - Implement __init__ with connection to Milvus URI
    - Implement create_collection() with full schema (id, user_id, memory_type, ts, chat_id, who, text, vector, hit_count, metadata)
    - Enable dynamic_field=True
    - Create AUTOINDEX for vector field with COSINE metric
    - Implement insert(), search(), query(), update(), delete() methods
    - _Requirements: 1.1, 1.2, 1.4_

  - [x] 2.4 Write unit tests for infrastructure clients


    - Test EmbeddingClient encode() returns correct dimension vectors
    - Test LLMClient chat_json() handles invalid JSON gracefully
    - Test MilvusStore CRUD operations
    - _Requirements: 1.1, 1.2, 1.3_

- [x] 3. Checkpoint - Ensure all tests pass








  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement memory processor modules




  - [x] 4.1 Implement EpisodicWriteDecider processor


    - Load EPISODIC_MEMORY_WRITE_FILTER prompt from prompts.py
    - Implement decide() method that calls LLM with conversation turns
    - Parse JSON response for write_episodic and records
    - Return WriteDecision dataclass
    - _Requirements: 2.1, 2.4, 2.5_
  - [x] 4.2 Write property tests for EpisodicWriteDecider



    - **Property 3: Chitchat and Knowledge Query Filtering**
    - **Validates: Requirements 2.4**
    - **Property 4: Explicit Remember Request Storage**
    - **Validates: Requirements 2.5**
  - [x] 4.3 Implement SemanticWriter processor


    - Load SEMANTIC_MEMORY_WRITER_PROMPT from prompts.py
    - Implement extract() method that analyzes episodic memory for facts
    - Parse JSON response for write_semantic and facts
    - Return SemanticExtraction dataclass
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 4.4 Implement EpisodicMerger processor

    - Load EPISODIC_MEMORY_MERGER_PROMPT from prompts.py
    - Implement merge() method that combines two episodic memories
    - Ensure merged record has source_chat_ids and merged_from_ids
    - Use earliest metadata.time as canonical time
    - _Requirements: 5.2, 5.4, 9.4, 9.5_

  - [x] 4.5 Write property tests for merge operations


    - **Property 10: Merge Threshold Enforcement**
    - **Validates: Requirements 5.2**
    - **Property 12: Merge Record Count Invariant**
    - **Validates: Requirements 5.4**
    - **Property 21: Merged Record Time Selection**
    - **Validates: Requirements 9.4**
    - **Property 22: Merged Record Source Tracking**
    - **Validates: Requirements 9.5**
  - [x] 4.6 Implement EpisodicSeparator processor


    - Load EPISODIC_MEMORY_SEPARATOR_PROMPT from prompts.py
    - Implement separate() method that rewrites two similar memories
    - Return tuple of updated memory dicts
    - _Requirements: 5.3, 5.5_

  - [x] 4.7 Write property test for separation


    - **Property 11: Separation Threshold Enforcement**
    - **Validates: Requirements 5.3**
  - [x] 4.8 Implement EpisodicReconsolidator processor


    - Load EPISODIC_MEMORY_RECONSOLIDATOR_PROMPTS from prompts.py
    - Implement reconsolidate() method with old_memory and current_context
    - Preserve metadata.time, chat_id, who, metadata.who
    - Append to metadata.updates array
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  - [x] 4.9 Write property tests for reconsolidation



    - **Property 8: Reconsolidation Field Preservation**
    - **Validates: Requirements 4.2**
    - **Property 9: Reconsolidation Updates Array Growth**
    - **Validates: Requirements 4.4**


- [x] 5. Checkpoint - Ensure all tests pass










  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Memory class core operations


  - [x] 6.1 Implement Memory class initialization


    - Use factory pattern to instantiate EmbeddingClient, LLMClient, MilvusStore
    - Initialize all processor modules
    - Create collection if not exists
    - _Requirements: 1.1, 8.6_
  - [x] 6.2 Implement Memory.add() method

    - Call EpisodicWriteDecider with conversation turns
    - Generate embeddings for qualifying records
    - Populate all required fields (user_id, ts, chat_id, memory_type, hit_count, metadata)
    - Insert into Milvus
    - Return list of created memory IDs
    - _Requirements: 2.1, 2.2, 2.3, 8.1_
  - [x] 6.3 Write property test for episodic memory completeness



    - **Property 2: Episodic Memory Field Completeness**
    - **Validates: Requirements 2.3**
  - [x] 6.4 Implement Memory.search() method

    - Generate embedding for query text
    - Search both episodic and semantic memories with user_id filter
    - Apply k_semantic and k_episodic limits
    - Rank results by similarity, memory_type, time decay, hit_count
    - Increment hit_count for retrieved memories
    - Return ranked MemoryRecord list
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [x] 6.5 Write property tests for search operations


    - **Property 5: Search Result Type Coverage**
    - **Validates: Requirements 3.2**
    - **Property 6: Search Result Limit Enforcement**
    - **Validates: Requirements 3.3**
    - **Property 7: Hit Count Increment on Retrieval**
    - **Validates: Requirements 3.5**
  - [x] 6.6 Implement Memory.update() method

    - Update specified memory record fields
    - Regenerate embedding if text changed
    - _Requirements: 8.3_

  - [x] 6.7 Implement Memory.delete() method
    - Delete specified memory record by ID

    - _Requirements: 8.4_
  - [x] 6.8 Implement Memory.reset() method
    - Delete all memories for specified user_id
    - Return count of deleted memories
    - _Requirements: 8.5_

  - [x] 6.9 Write property test for reset operation


    - **Property 17: Reset Operation Completeness**
    - **Validates: Requirements 8.5**

- [x] 7. Checkpoint - Ensure all tests pass




  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement consolidation flow




  - [x] 8.1 Implement similarity search for consolidation


    - For each episodic memory, find Top-N similar candidates
    - Calculate cosine similarity scores
    - Apply T_merge_high (0.85) and T_amb_low (0.65) thresholds
    - _Requirements: 5.1_
  - [x] 8.2 Implement merge constraint checking


    - Check who field equality
    - Check time constraints: 30 min for same chat_id, 7 days for different
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 8.3 Write property tests for merge constraints


    - **Property 18: Same Chat Merge Time Constraint**
    - **Validates: Requirements 9.1**
    - **Property 19: Different Chat Merge Time Constraint**
    - **Validates: Requirements 9.2**
    - **Property 20: Who Field Merge Constraint**
    - **Validates: Requirements 9.3**
  - [x] 8.4 Implement Memory.consolidate() method


    - Iterate through all episodic memories for user
    - Apply merge logic for high similarity pairs
    - Apply separation logic for medium similarity pairs
    - Call SemanticWriter for each episodic memory
    - Create semantic memories for extracted facts
    - Return ConsolidationStats
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.6, 7.1_

  - [x] 8.5 Write property test for semantic memory creation


    - **Property 13: Semantic Memory Field Completeness**
    - **Validates: Requirements 6.6**

- [x] 9. Implement reconsolidation integration






  - [x] 9.1 Integrate reconsolidation into search flow




    - After search returns results, identify used episodic memories
    - Call EpisodicReconsolidator with current context
    - Update memory records with reconsolidated content
    - Regenerate embeddings for updated records
    - _Requirements: 4.1, 4.5_

- [x] 10. Implement logging and error handling




  - [x] 10.1 Add logging throughout the system


    - Log memory operations with type, user_id, affected count
    - Log consolidation statistics
    - _Requirements: 10.4, 10.5_

  - [x] 10.2 Implement custom exception classes

    - MilvusConnectionError with URI and original error
    - OpenRouterError with model, attempts, and last error
    - _Requirements: 10.1, 10.2_

- [x] 11. Final Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Create integration tests







  - [x] 12.1 Write end-to-end integration tests





    - Test full add -> search -> reconsolidate flow
    - Test consolidation with merge and separation
    - Test semantic memory extraction
    - _Requirements: All_
