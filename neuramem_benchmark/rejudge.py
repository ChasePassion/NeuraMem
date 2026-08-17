"""Re-judge LoCoMo CSV rows (async port of rejudge.py).

Differs from judge.py: uses complete() + manual JSON/keyword parsing
(the W3-grade path) and by default re-grades ungraded OR previously
WRONG rows.
"""

import argparse
import asyncio
import csv
import json
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


async def judge_row(llm: OpenAILLM, row: dict) -> tuple[str, str]:
    category_num = (
        int(row.get("category", "1")) if str(row.get("category", "")).isdigit() else 1
    )
    gold = preprocess_answer(category_num, row.get("answer", ""))
    prompt = get_judge_prompt(
        category=category_num,
        question=row.get("question", ""),
        answer=gold,
        response=row.get("response", ""),
    )
    try:
        resp = await llm.complete(JUDGE_SYSTEM_PROMPT, prompt)
        raw = resp.content or ""
        clean = raw.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
            clean = clean.split("```")[0].strip()
        think_open = chr(60) + "think" + chr(62)
        think_close = chr(60) + "/think" + chr(62)
        if think_open in clean and think_close in clean:
            clean = clean.split(think_close, 1)[-1].strip()
        json_start, json_end = clean.find("{"), clean.rfind("}")
        if json_start != -1 and json_end > json_start:
            clean = clean[json_start:json_end + 1]
        try:
            parsed = json.loads(clean)
            label = str(parsed.get("label", "")).strip().upper()
            reasoning = str(parsed.get("reasoning", "")).strip()
        except json.JSONDecodeError:
            upper_raw = raw.upper()
            if '"CORRECT"' in upper_raw or (
                "CORRECT" in upper_raw and "WRONG" not in upper_raw
            ):
                label = "CORRECT"
            else:
                label = "WRONG"
            reasoning = raw[:300]
    except Exception as e:  # noqa: BLE001
        return "WRONG", f"Judge call failed: {e}"
    return ("CORRECT", reasoning) if "CORRECT" in label else ("WRONG", reasoning)


async def run(args) -> None:
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    config = build_benchmark_config()
    llm = OpenAILLM(config.llm)

    to_grade = [
        i for i, r in enumerate(rows)
        if args.force or not r.get("result") or r.get("result") == "WRONG"
    ]
    logger.info("Total rows: %d, rows to re-judge: %d", len(rows), len(to_grade))

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.threads)
    correct = 0

    async def process(idx: int) -> str:
        nonlocal correct
        row = rows[idx]
        async with semaphore:
            label, reasoning = await judge_row(llm, row)
            row["result"] = label
            row["reasoning"] = reasoning
        async with write_lock:
            with open(args.input, "w", encoding="utf-8", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        if label == "CORRECT":
            correct += 1
        return label

    results = await asyncio.gather(*(process(i) for i in to_grade))
    for _ in tqdm(results, total=len(results), desc="Re-judging"):
        pass
    total = len(results)
    if total:
        logger.info("Re-judge complete: %d/%d CORRECT (%.1f%%)",
                    correct, total, 100 * correct / total)
    else:
        logger.info("No rows graded")


def main():
    parser = argparse.ArgumentParser(description="Re-judge LoCoMo result CSV")
    parser.add_argument("--input", default="result/locomo_neuramem_results.csv")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
