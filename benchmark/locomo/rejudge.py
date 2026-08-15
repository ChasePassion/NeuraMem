"""Re-judge existing LoCoMo result CSV using llm_client.chat() + JSON parsing.
Run from project root:
    python benchmark/locomo/rejudge.py --input result/locomo_neuramem_results.csv
"""
import argparse
import csv
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import dotenv
dotenv.load_dotenv()

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.memory_system.clients import LLMClient
from src.memory_system.config import MemoryConfig
from benchmark.locomo.locomo_prompts import get_judge_prompt, JUDGE_SYSTEM_PROMPT, preprocess_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def judge_row(llm_client: LLMClient, row: dict) -> tuple[str, str]:
    category_num = int(row.get("category", "1")) if str(row.get("category", "")).isdigit() else 1
    gold = preprocess_answer(category_num, row.get("answer", ""))
    prompt = get_judge_prompt(
        category=category_num,
        question=row.get("question", ""),
        answer=gold,
        response=row.get("response", ""),
    )
    try:
        raw = llm_client.chat(system_prompt=JUDGE_SYSTEM_PROMPT, user_message=prompt)
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.split("```")[0].strip()
        try:
            parsed = json.loads(clean)
            label = str(parsed.get("label", "")).strip().upper()
            reasoning = str(parsed.get("reasoning", "")).strip()
        except json.JSONDecodeError:
            upper_raw = raw.upper()
            if '"CORRECT"' in upper_raw or ("CORRECT" in upper_raw and "WRONG" not in upper_raw):
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
    parser = argparse.ArgumentParser(description="Re-judge LoCoMo result CSV")
    parser.add_argument("--input", default="result/locomo_neuramem_results.csv")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Re-grade all rows, even already graded")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    config = MemoryConfig()
    llm_client = LLMClient(
        api_key=config.llm_primary_api_key,
        base_url=config.llm_primary_base_url,
        model=config.llm_primary_model,
        fallback_api_key=config.llm_fallback_api_key,
        fallback_base_url=config.llm_fallback_base_url,
    )

    to_grade = [i for i, r in enumerate(rows) if args.force or not r.get("result") or r.get("result") == "WRONG"]
    logger.info(f"Total rows: {len(rows)}, rows to re-judge: {len(to_grade)}")

    file_lock = threading.Lock()
    correct = 0
    wrong = 0

    def process(idx: int):
        row = rows[idx]
        label, reasoning = judge_row(llm_client, row)
        row["result"] = label
        row["reasoning"] = reasoning
        with file_lock:
            with open(args.input, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        return label

    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(process, i): i for i in to_grade}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Re-judging"):
            try:
                result = fut.result()
                if result == "CORRECT":
                    correct += 1
                else:
                    wrong += 1
            except Exception as e:
                logger.error(f"Error: {e}")
                wrong += 1

    total = correct + wrong
    logger.info(f"\nRe-judge complete: {correct}/{total} CORRECT ({100*correct/total:.1f}%)" if total else "No rows graded")


if __name__ == "__main__":
    main()
