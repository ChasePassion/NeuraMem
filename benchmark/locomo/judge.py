"""Standalone LLM Judge for LoCoMo evaluation CSVs.
"""

import argparse
import csv
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import dotenv
dotenv.load_dotenv()

from src.memory_system.clients import LLMClient
from src.memory_system.config import MemoryConfig

try:
    from benchmark.locomo.llm_config import apply_minimax_primary
except ImportError:
    from llm_config import apply_minimax_primary

try:
    from benchmark.locomo.locomo_prompts import (
        get_judge_prompt,
        JUDGE_SYSTEM_PROMPT,
        preprocess_answer,
    )
except ImportError:
    from locomo_prompts import (
        get_judge_prompt,
        JUDGE_SYSTEM_PROMPT,
        preprocess_answer,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Judge ungraded responses in LoCoMo CSV")
    parser.add_argument("--input", default="result/locomo_neuramem_results.csv", help="Path to result CSV")
    parser.add_argument("--threads", type=int, default=8, help="Concurrency for LLM judge")
    parser.add_argument("--force", action="store_true", help="Re-grade already graded rows")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    config = MemoryConfig()
    apply_minimax_primary(config)
    llm_client = LLMClient(
        api_key=config.llm_primary_api_key,
        base_url=config.llm_primary_base_url,
        model=config.llm_primary_model,
    )

    to_grade_indices = [
        i for i, r in enumerate(rows)
        if args.force or not r.get("result")
    ]
    logger.info(f"Total rows: {len(rows)}, to grade: {len(to_grade_indices)}")

    file_lock = threading.Lock()

    def grade_row(idx: int):
        row = rows[idx]
        category_num = int(row.get("category", "1")) if str(row.get("category", "")).isdigit() else 1
        gold = preprocess_answer(category_num, row.get("answer", ""))
        prompt = get_judge_prompt(
            category=category_num,
            question=row.get("question", ""),
            answer=gold,
            response=row.get("response", ""),
        )
        res = llm_client.chat_json(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_message=prompt,
            default={"label": "WRONG", "reasoning": "Judge failed"}
        )
        label = "CORRECT" if "CORRECT" in str(res.get("label", "")).upper() else "WRONG"
        row["result"] = label
        row["reasoning"] = str(res.get("reasoning", ""))

        with file_lock:
            with open(args.input, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(grade_row, idx) for idx in to_grade_indices]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            try:
                f.result()
            except Exception as e:
                logger.error(f"Grading error: {e}")

    logger.info("Judging complete!")


if __name__ == "__main__":
    main()
