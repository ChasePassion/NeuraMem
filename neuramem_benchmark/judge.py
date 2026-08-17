"""Standalone LLM judge for LoCoMo CSVs (async port of judge.py).

Grades ungraded rows using complete_json; rewrites the CSV after each
graded row under a file lock. CSV columns unchanged.
"""

import argparse
import asyncio
import csv
import logging
import os

from tqdm import tqdm

import dotenv

from neuramem.llm.openai_adapter import OpenAILLM
from neuramem_benchmark.llm_config import build_benchmark_config
from neuramem_benchmark.locomo_prompts import (
    get_judge_prompt,
    JUDGE_SYSTEM_PROMPT,
    preprocess_answer,
)

dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def run(args) -> None:
    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    config = build_benchmark_config()
    llm = OpenAILLM(config.llm)

    to_grade = [i for i, r in enumerate(rows) if args.force or not r.get("result")]
    logger.info("Total rows: %d, to grade: %d", len(rows), len(to_grade))

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.threads)

    async def grade_row(idx: int):
        row = rows[idx]
        async with semaphore:
            category_num = (
                int(row.get("category", "1"))
                if str(row.get("category", "")).isdigit()
                else 1
            )
            gold = preprocess_answer(category_num, row.get("answer", ""))
            prompt = get_judge_prompt(
                category=category_num,
                question=row.get("question", ""),
                answer=gold,
                response=row.get("response", ""),
            )
            res = await llm.complete_json(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_message=prompt,
                default={"label": "WRONG", "reasoning": "Judge failed"},
            )
            data = res.parsed_data or {}
            row["result"] = (
                "CORRECT"
                if "CORRECT" in str(data.get("label", "")).upper()
                else "WRONG"
            )
            row["reasoning"] = str(data.get("reasoning", ""))
        async with write_lock:
            with open(args.input, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    await tqdm(asyncio.gather(*(grade_row(i) for i in to_grade)), total=len(to_grade), desc="Judging")  # type: ignore[arg-type]
    logger.info("Judging complete!")


def main():
    parser = argparse.ArgumentParser(description="Judge ungraded responses in LoCoMo CSV")
    parser.add_argument("--input", default="result/locomo_neuramem_results.csv")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
