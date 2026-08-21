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
from neuramem_benchmark.grading import judge_row
from neuramem_benchmark.llm_config import build_benchmark_config

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

    to_grade = [
        i
        for i, row in enumerate(rows)
        if args.force
        or not row.get("result")
        or (getattr(args, "include_wrong", False) and row.get("result") == "WRONG")
    ]
    logger.info("Total rows: %d, to grade: %d", len(rows), len(to_grade))

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.threads)

    async def grade_row(idx: int):
        row = rows[idx]
        async with semaphore:
            row["result"], row["reasoning"], _ = await judge_row(llm, row)
        async with write_lock:
            with open(args.input, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    await tqdm(asyncio.gather(*(grade_row(i) for i in to_grade)), total=len(to_grade), desc="Judging")  # type: ignore[arg-type]
    logger.info("Judging complete!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge responses in a LoCoMo CSV")
    parser.add_argument("--input", default="result/locomo_neuramem_results.csv")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-wrong",
        action="store_true",
        help="Also re-judge rows currently labeled WRONG",
    )
    return parser


def main():
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
