"""LoCoMo benchmark evaluation runner for NeuraMem.

Runs memory retrieval, answer generation, and records evaluation metrics to CSV.
"""

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.memory_system import Memory, MemoryConfig
from src.memory_system.clients import LLMClient

try:
    from benchmark.locomo.llm_config import apply_minimax_primary
except ImportError:
    from llm_config import apply_minimax_primary

try:
    from benchmark.locomo.locomo_prompts import (
        get_answer_generation_prompt,
        get_judge_prompt,
        JUDGE_SYSTEM_PROMPT,
        preprocess_answer,
    )
except ImportError:
    from locomo_prompts import (
        get_answer_generation_prompt,
        get_judge_prompt,
        JUDGE_SYSTEM_PROMPT,
        preprocess_answer,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QA_FIELDNAMES = [
    "sample_index",
    "sample_id",
    "question_id",
    "category",
    "question",
    "answer",
    "response",
    "evidence",
    "retrieved_count",
    "time_cost",
    "result",
    "reasoning",
    "timestamp",
]


def load_locomo_qa_list(path: str, sample_idx: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load QA pairs from LoCoMo JSON file."""
    if not os.path.exists(path):
        alt_path = Path("E:/code/locomo/data/locomo10.json")
        if alt_path.exists():
            path = str(alt_path)
        else:
            raise FileNotFoundError(f"LoCoMo dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_list = []
    for s_idx, sample in enumerate(data):
        if sample_idx is not None and s_idx != sample_idx:
            continue
        sample_id = sample.get("sample_id", f"sample_{s_idx}")
        user_id = f"sample_{s_idx}"
        for q_idx, qa in enumerate(sample.get("qa", [])):
            # Skip adversarial category 5, same as OpenViking's run_eval:
            # these questions test hallucination (asked about events that
            # never happened) and are excluded from the score.
            if str(qa.get("category", "")) == "5":
                continue
            qa_list.append({
                "sample_index": s_idx,
                "sample_id": sample_id,
                "user_id": user_id,
                "question_id": f"{sample_id}_q{q_idx}",
                "category": str(qa.get("category", "")),
                "question": qa.get("question", ""),
                "answer": str(qa.get("answer", "")),
                "evidence": json.dumps(qa.get("evidence", []), ensure_ascii=False),
            })
    return qa_list


def answer_question(
    memory: Memory,
    llm_client: LLMClient,
    qa_item: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute search and answer generation for a single QA item."""
    user_id = qa_item["user_id"]
    question = qa_item["question"]
    category = qa_item["category"]

    start_time = time.time()

    # 1. Search memory context
    search_res = memory.search(query=question, user_id=user_id)
    episodic = search_res.get("episodic", [])
    semantic = search_res.get("semantic", [])
    all_memories = episodic + semantic

    # 2. Build prompt
    prompt = get_answer_generation_prompt(
        question=question,
        search_results=all_memories,
        reference_date="2023",
    )

    # 3. Call LLM to generate answer
    raw_response = llm_client.chat(
        system_prompt="You are a helpful assistant answering questions about past conversations accurately.",
        user_message=prompt,
    )

    # Strip reasoning-think block emitted by reasoning models (e.g. MiniMax-M3)
    think_open = chr(60) + "think" + chr(62)
    think_close = chr(60) + "/think" + chr(62)
    if think_open in raw_response and think_close in raw_response:
        raw_response = raw_response.split(think_close, 1)[-1].strip()

    # Extract final answer after "ANSWER:" if present
    if "ANSWER:" in raw_response:
        final_answer = raw_response.split("ANSWER:")[-1].strip()
    else:
        final_answer = raw_response.strip()

    # 4. Judge which retrieved memories were actually used in the answer, then
    #    assign them to narrative groups. Mirrors the demo's full loop:
    #    search -> respond -> judge usage -> reconsolidate (narrative grouping).
    #    Later searches then expand these groups, exercising narrative memory.
    try:
        used_texts = memory._memory_usage_judge.judge_used_memories(
            episodic_memories=[mem.text for mem in episodic],
            last_user=question,
            last_assistant=final_answer,
        )
        used_ids = [mem.id for mem in episodic if mem.text in used_texts]
        if used_ids:
            assignments = memory.assign_to_narrative_group(used_ids, user_id)
            logger.info(
                f"Assigned {len(assignments)} episodic memories to narrative groups "
                f"for {qa_item['question_id']}"
            )
    except Exception as e:  # noqa: BLE001 - never fail the eval over this
        logger.warning(f"Usage judge / narrative assignment failed for {qa_item['question_id']}: {e}")

    elapsed = round(time.time() - start_time, 2)

    return {
        "sample_index": qa_item["sample_index"],
        "sample_id": qa_item["sample_id"],
        "question_id": qa_item["question_id"],
        "category": category,
        "question": question,
        "answer": qa_item["answer"],
        "response": final_answer,
        "evidence": qa_item["evidence"],
        "retrieved_count": len(all_memories),
        "time_cost": elapsed,
        "result": "",
        "reasoning": "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def judge_single_response(
    llm_client: LLMClient,
    row: Dict[str, Any],
) -> tuple[str, str]:
    """Judge a single question answer using LLM-as-a-Judge."""
    import json as _json

    category_num = int(row["category"]) if str(row["category"]).isdigit() else 1
    gold_answer = preprocess_answer(category_num, row["answer"])
    prompt = get_judge_prompt(
        category=category_num,
        question=row["question"],
        answer=gold_answer,
        response=row["response"],
    )

    try:
        raw = llm_client.chat(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=prompt,
        )
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.split("```")[0].strip()
        # Strip reasoning-think block emitted by reasoning models (e.g. MiniMax-M3)
        think_open = chr(60) + "think" + chr(62)
        think_close = chr(60) + "/think" + chr(62)
        if think_open in clean and think_close in clean:
            clean = clean.split(think_close, 1)[-1].strip()
        # Extract the JSON object wherever it appears
        json_start = clean.find("{")
        json_end = clean.rfind("}")
        if json_start != -1 and json_end > json_start:
            clean = clean[json_start:json_end + 1]
        try:
            parsed = _json.loads(clean)
            label = str(parsed.get("label", "")).strip().upper()
            reasoning = str(parsed.get("reasoning", "")).strip()
        except _json.JSONDecodeError:
            # Fallback: scan raw text for CORRECT / WRONG keyword
            upper_raw = raw.upper()
            if "\"CORRECT\"" in upper_raw or ("CORRECT" in upper_raw and "WRONG" not in upper_raw):
                label = "CORRECT"
            else:
                label = "WRONG"
            reasoning = raw[:300]
    except Exception as e:
        return "WRONG", f"Judge call failed: {e}"

    if "CORRECT" in label:
        return "CORRECT", reasoning
    return "WRONG", reasoning



def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuraMem on LoCoMo dataset")
    parser.add_argument("--input", default="data/locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--output", default="result/locomo_neuramem_results.csv", help="Path to output CSV")
    parser.add_argument("--sample", type=int, default=None, help="Sample index (0-9) to evaluate")
    parser.add_argument("--question-index", type=int, default=None, help="Single question index for smoke test")
    parser.add_argument("--count", type=int, default=None, help="Max QA items to process")
    parser.add_argument("--threads", type=int, default=4, help="Parallel worker threads")
    parser.add_argument("--milvus-uri", default=None, help="Milvus URI")
    parser.add_argument("--judge", action="store_true", default=True, help="Auto-judge responses right after answering")
    parser.add_argument("--no-judge", dest="judge", action="store_false", help="Skip auto-judging")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    config = MemoryConfig()
    if args.milvus_uri:
        config.milvus_uri = args.milvus_uri
    elif not config.milvus_uri:
        config.milvus_uri = os.getenv("MILVUS_URL", "http://117.72.161.187:19530")

    apply_minimax_primary(config)

    memory = Memory(config)
    llm_client = memory._llm_client

    qa_list = load_locomo_qa_list(args.input, args.sample)
    if args.question_index is not None and args.sample is not None:
        qa_list = [q for i, q in enumerate(qa_list) if i == args.question_index]
    elif args.count is not None:
        qa_list = qa_list[:args.count]

    logger.info(f"Loaded {len(qa_list)} questions for evaluation. Output: {args.output}")

    # Prepare CSV file
    csv_file = open(args.output, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=QA_FIELDNAMES)
    writer.writeheader()
    csv_file.flush()

    write_lock = threading.Lock()
    evaluated_rows = []

    def process_item(item):
        row = answer_question(memory, llm_client, item)
        if args.judge:
            label, reasoning = judge_single_response(llm_client, row)
            row["result"] = label
            row["reasoning"] = reasoning

        with write_lock:
            writer.writerow(row)
            csv_file.flush()
            evaluated_rows.append(row)
        return row

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_item, item) for item in qa_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Evaluating QA"):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Error processing question: {e}")

    csv_file.close()
    elapsed = time.time() - start_time
    logger.info(f"Evaluation complete! Processed {len(evaluated_rows)} questions in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
