"""LoCoMo eval runner (async two-phase port of run_eval.py).

Closed loop now goes through the public API: search_async -> own answer
call -> report_usage_async (judge + narrative assignment inside the
facade). The legacy text->id matching and _memory_usage_judge reach-in
are gone. Concurrency switched from threads to asyncio semaphore
(contextvars scope attribution is task-safe; UsageStats.scope() gives
per-question deltas directly — no before/after snapshots needed).

NEW: refuses to evaluate samples without an ingest manifest (s8 lesson)
and records an evidence-recall@k column (metrics.py, OpenViking pointer
resolution).
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import dotenv

from neuramem.config import MemoryConfig
from neuramem.llm.openai_adapter import OpenAILLM, UsageStats
from neuramem.memory import Memory
from neuramem_benchmark.llm_config import build_benchmark_config
from neuramem_benchmark.locomo import load_locomo_qa_list, load_locomo_samples
from neuramem_benchmark.locomo_prompts import (
    get_answer_generation_prompt,
    get_judge_prompt,
    JUDGE_SYSTEM_PROMPT,
    preprocess_answer,
)
from neuramem_benchmark.metrics import evidence_recall, resolve_evidence_texts

dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about past "
    "conversations accurately."
)

QA_FIELDNAMES = [
    "sample_index",
    "sample_id",
    "question_id",
    "category",
    "question",
    "answer",
    "response",
    "evidence",
    "evidence_recall",
    "retrieved_count",
    "time_cost",
    "result",
    "reasoning",
    "timestamp",
    "cache_hit_tokens",
    "cache_prompt_tokens",
    "answer_cache_hit_tokens",
    "answer_cache_prompt_tokens",
    "memory_cache_hit_tokens",
    "memory_cache_prompt_tokens",
]


def _strip_reasoning(raw: str) -> str:
    think_open = chr(60) + "think" + chr(62)
    think_close = chr(60) + "/think" + chr(62)
    if think_open in raw and think_close in raw:
        raw = raw.split(think_close, 1)[-1].strip()
    if "ANSWER:" in raw:
        return raw.split("ANSWER:")[-1].strip()
    return raw.strip()


async def judge_response(llm: OpenAILLM, row: Dict[str, Any]) -> tuple[str, str]:
    """LLM-as-judge for one row (lenient rubric, W3 prompt verbatim)."""
    category_num = int(row["category"]) if str(row["category"]).isdigit() else 1
    gold = preprocess_answer(category_num, row["answer"])
    prompt = get_judge_prompt(
        category=category_num,
        question=row["question"],
        answer=gold,
        response=row["response"],
    )
    try:
        resp = await llm.complete(
            JUDGE_SYSTEM_PROMPT, prompt, call_label="judge"
        )
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
    except Exception as e:  # noqa: BLE001 - judge failure scores WRONG
        return "WRONG", f"Judge call failed: {e}"
    return ("CORRECT", reasoning) if "CORRECT" in label else ("WRONG", reasoning)


async def answer_question(
    memory: Optional[Memory],
    llm: OpenAILLM,
    qa_item: Dict[str, Any],
    evidence_texts: List[str],
    judge_on: bool,
    no_memory: bool = False,
) -> Dict[str, Any]:
    """One QA through the two-phase loop, inside a fresh usage scope.

    no_memory=True is the baseline arm of the memory-uplift comparison:
    same QA, same answer template (empty memory list), same judge — no
    retrieval and no closed loop, so the accuracy delta against the
    memory arm measures what the memory system itself contributes.
    """
    user_id = qa_item["user_id"]
    question = qa_item["question"]
    start_time = time.time()

    with llm.usage_stats.scope():
        if no_memory:
            result = None
            all_memories = []
            recall = None
        else:
            # Phase 1: retrieval (correlation token)
            result = await memory.search_async(question, user_id)
            episodic, semantic = result.episodic, result.semantic
            all_memories = episodic + semantic

            recall = evidence_recall(
                [m.text for m in all_memories], evidence_texts
            )

        # Phase 2a: answer generation (runner-owned call, label "answer")
        prompt = get_answer_generation_prompt(
            question=question,
            search_results=all_memories,
            reference_date="2023",
        )
        resp = await llm.complete(
            ANSWER_SYSTEM_PROMPT, prompt, call_label="answer"
        )
        final_answer = _strip_reasoning(resp.content or "")

        # Phase 2b: closed loop via the public facade API
        if result is not None:
            report = await memory.report_usage_async(result, final_answer)
            if report.assignments:
                logger.info(
                    "Assigned %d episodic memories to narrative groups for %s",
                    len(report.assignments), qa_item["question_id"],
                )

        row: Dict[str, Any] = {
            "sample_index": qa_item["sample_index"],
            "sample_id": qa_item["sample_id"],
            "question_id": qa_item["question_id"],
            "category": qa_item["category"],
            "question": question,
            "answer": qa_item["answer"],
            "response": final_answer,
            "evidence": qa_item["evidence"],
            "evidence_recall": "" if recall is None else str(int(recall)),
            "retrieved_count": len(all_memories),
            "time_cost": round(time.time() - start_time, 2),
            "result": "",
            "reasoning": "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if judge_on:
            row["result"], row["reasoning"] = await judge_response(llm, row)

        # scope snapshots ARE this question's deltas (fresh scope per task)
        total = llm.usage_stats.scope_snapshot()
        answer_s = llm.usage_stats.scope_snapshot("answer")
        usage_judge_s = llm.usage_stats.scope_snapshot("usage_judge")
        row["cache_hit_tokens"] = total["cache_read_tokens"]
        row["cache_prompt_tokens"] = (
            total["input_tokens"]
            + total["cache_read_tokens"]
            + total["cache_write_tokens"]
        )
        row["answer_cache_hit_tokens"] = answer_s["cache_read_tokens"]
        row["answer_cache_prompt_tokens"] = (
            answer_s["input_tokens"]
            + answer_s["cache_read_tokens"]
            + answer_s["cache_write_tokens"]
        )
        # memory-system scoped slice: answer + usage_judge (judge excluded)
        row["memory_cache_hit_tokens"] = (
            answer_s["cache_read_tokens"] + usage_judge_s["cache_read_tokens"]
        )
        row["memory_cache_prompt_tokens"] = (
            answer_s["input_tokens"] + answer_s["cache_read_tokens"]
            + answer_s["cache_write_tokens"]
            + usage_judge_s["input_tokens"] + usage_judge_s["cache_read_tokens"]
            + usage_judge_s["cache_write_tokens"]
        )
    return row


def _check_manifests(qa_list: List[Dict[str, Any]], manifest_dir: str) -> None:
    """Refuse to eval samples whose ingest manifest is missing (s8 lesson)."""
    missing = []
    for s_idx in sorted({q["sample_index"] for q in qa_list}):
        path = os.path.join(manifest_dir, f"ingest_manifest_{s_idx}.json")
        if not os.path.exists(path):
            missing.append(s_idx)
    if missing:
        raise SystemExit(
            f"Ingest manifest missing for sample(s) {missing} in "
            f"'{manifest_dir}' — the ingest may have been truncated "
            f"(s8 lesson, RUN_RECORD 8.1). Re-run ingest for those samples "
            f"or pass --no-manifest-check to override."
        )


async def run(args) -> None:
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    qa_list = load_locomo_qa_list(args.input, args.sample)
    if args.question_index is not None and args.sample is not None:
        qa_list = [q for i, q in enumerate(qa_list) if i == args.question_index]
    elif args.count is not None:
        qa_list = qa_list[: args.count]

    if args.no_memory:
        logger.info("no-memory baseline mode: no retrieval, no closed loop")
    elif not args.no_manifest_check:
        _check_manifests(qa_list, args.manifest_dir)

    # evidence texts keyed by question_id: robust to any qa filtering
    # (--count / --question-index), OpenViking pointer resolution
    evidence_by_qid: Dict[str, List[str]] = {}
    for sample in load_locomo_samples(args.input, args.sample):
        sample_id = sample.get("sample_id", f"sample_{sample['sample_index']}")
        for q_idx, qa in enumerate(sample.get("qa", [])):
            if str(qa.get("category", "")) == "5":
                continue
            evidence_by_qid[f"{sample_id}_q{q_idx}"] = resolve_evidence_texts(
                sample, qa.get("evidence", [])
            )

    config = build_benchmark_config(args.milvus_uri)
    llm = OpenAILLM(config.llm)  # shared with the facade: one usage aggregate
    memory = None if args.no_memory else Memory(config, llm=llm)

    logger.info("Loaded %d questions for evaluation. Output: %s",
                len(qa_list), args.output)

    csv_file = open(args.output, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=QA_FIELDNAMES)
    writer.writeheader()
    csv_file.flush()

    semaphore = asyncio.Semaphore(args.threads)
    evaluated = 0
    start_time = time.time()

    async def worker(qa_item: Dict[str, Any]):
        nonlocal evaluated
        async with semaphore:
            try:
                row = await answer_question(
                    memory,
                    llm,
                    qa_item,
                    evidence_by_qid.get(qa_item["question_id"], []),
                    args.judge,
                    no_memory=args.no_memory,
                )
            except Exception as e:  # noqa: BLE001 - one question must not kill the run
                logger.error("Error processing %s: %s", qa_item["question_id"], e)
                return
            writer.writerow(row)
            csv_file.flush()
            evaluated += 1

    tasks = [asyncio.create_task(worker(q)) for q in qa_list]
    for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating QA"):
        await _

    csv_file.close()
    elapsed = time.time() - start_time
    logger.info("Evaluation complete! Processed %d questions in %.1fs",
                evaluated, elapsed)

    # KV cache summary by call type (column semantics identical to W3)
    def _rate_line(name: str, snapshot: Dict[str, Any]) -> str:
        rate = UsageStats.hit_rate_of(snapshot)
        prompt = (
            snapshot["input_tokens"] + snapshot["cache_read_tokens"]
            + snapshot["cache_write_tokens"]
        )
        if rate is None:
            return f"{name}: no cache info (prompt={prompt} tokens, {snapshot['calls']} calls)"
        return (f"{name}: hit={snapshot['cache_read_tokens']}/{prompt} tokens "
                f"over {snapshot['calls']} calls -> {rate:.2%}")

    stats = llm.usage_stats
    answer_total = stats.snapshot("answer")
    usage_judge_total = stats.snapshot("usage_judge")
    merged = {
        k: answer_total.get(k, 0) + usage_judge_total.get(k, 0)
        for k in ("calls", "input_tokens", "output_tokens", "cache_read_tokens",
                  "cache_write_tokens", "reasoning_tokens", "total_tokens", "cost")
    }
    logger.info("KV cache (prefix cache) hit rates by call type:")
    logger.info("  " + _rate_line("memory-system eval (usage_judge+answer)", merged))
    logger.info("  " + _rate_line("  - answer      (memory-RAG generation)", answer_total))
    logger.info("  " + _rate_line("  - usage_judge (memory usage check)", usage_judge_total))
    logger.info("  " + _rate_line("judge       (eval tool, excluded)", stats.snapshot("judge")))
    logger.info("  " + _rate_line("overall     (all calls)", stats.snapshot()))


def main():
    parser = argparse.ArgumentParser(description="Evaluate NeuraMem on LoCoMo")
    parser.add_argument("--input", default="data/locomo10.json")
    parser.add_argument("--output", default="result/locomo_neuramem_results.csv")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--question-index", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4,
                        help="Concurrency (asyncio semaphore limit)")
    parser.add_argument("--milvus-uri", default=None)
    parser.add_argument("--judge", action="store_true", default=True)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--manifest-dir", default="result")
    parser.add_argument("--no-manifest-check", action="store_true")
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Baseline arm: answer without any memory injection (uplift comparison)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
