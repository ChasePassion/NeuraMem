"""One-Click LoCoMo Benchmark Runner for NeuraMem.

Executes the full pipeline: Ingest -> QA Eval -> Auto Judge -> Metric Breakdown Report.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON_EXEC = r"E:\Anaconda\envs\Langchain_learn\python.exe"


def run_cmd(cmd_list: list[str]) -> bool:
    """Run command synchronously and stream output."""
    logger.info(f"Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, shell=False)
    return res.returncode == 0


def count_samples(data_path: str) -> int:
    """Count samples in the LoCoMo JSON file (0 if unreadable)."""
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            return len(json.load(f))
    except Exception:  # noqa: BLE001 - fall back to serial ingest
        logger.warning(f"Cannot read {data_path} to count samples; using serial ingest")
        return 0


def parallel_ingest(args) -> None:
    """Ingest each sample in its own subprocess, up to --ingest-parallel at once.

    Samples map to distinct Milvus user_ids, so they are fully independent:
    parallel workers divide the ingest wall-clock time roughly by the worker
    count, at the cost of higher API concurrency (DeepSeek/SiliconFlow).
    """
    n_samples = count_samples(args.data)
    if n_samples <= 1:
        cmd = [
            PYTHON_EXEC,
            "benchmark/locomo/import_to_neuramem.py",
            "--input", args.data,
        ]
        if args.max_sessions is not None:
            cmd.extend(["--max-sessions", str(args.max_sessions)])
        if not run_cmd(cmd):
            logger.error("Ingest failed!")
            sys.exit(1)
        return

    workers = min(args.ingest_parallel, n_samples)
    logger.info(f"Parallel ingest: {workers} worker(s) across {n_samples} samples")

    def ingest_one(sample_idx: int) -> bool:
        cmd = [
            PYTHON_EXEC,
            "benchmark/locomo/import_to_neuramem.py",
            "--input", args.data,
            "--sample", str(sample_idx),
        ]
        if args.max_sessions is not None:
            cmd.extend(["--max-sessions", str(args.max_sessions)])
        return run_cmd(cmd)

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ingest_one, i): i for i in range(n_samples)}
        for f in as_completed(futures):
            sample_idx = futures[f]
            try:
                ok = f.result()
            except Exception as e:  # noqa: BLE001 - report and continue
                ok = False
                logger.error(f"Sample {sample_idx} ingest raised: {e}")
            if not ok:
                failed.append(sample_idx)
                logger.error(f"Sample {sample_idx} ingest failed")

    if failed:
        logger.error(f"Ingest failed for samples: {sorted(failed)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="End-to-end LoCoMo Benchmark for NeuraMem")
    parser.add_argument("--data", default="data/locomo10.json", help="Path to locomo10.json dataset")
    parser.add_argument("--output", default="result/locomo_neuramem_results.csv", help="Path to result CSV")
    parser.add_argument("--sample", type=int, default=0, help="Sample index (0-9). Default: 0 for standard test")
    parser.add_argument("--all-samples", action="store_true", help="Run all 10 conversation samples")
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit number of sessions per sample for quick testing")
    parser.add_argument("--qa-count", type=int, default=None, help="Limit number of QA items")
    parser.add_argument("--threads", type=int, default=4, help="Concurrency for QA evaluation")
    parser.add_argument(
        "--ingest-parallel",
        type=int,
        default=4,
        help=(
            "Parallel sample-level ingest subprocesses (only for --all-samples). "
            "Default: 4. Use 1 for serial ingest."
        ),
    )
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest step if already ingested")
    args = parser.parse_args()

    sample_arg = None if args.all_samples else args.sample

    print("=" * 80)
    print("      Starting NeuraMem LoCoMo Benchmark (Same Settings as OpenViking)     ")
    print(f"      Dataset: {args.data} | Sample: {'ALL' if args.all_samples else sample_arg}    ")
    print("=" * 80)

    # 1. Ingest
    if not args.skip_ingest:
        logger.info("=== STEP 1: Ingesting Conversation Dialogue into NeuraMem ===")
        if args.ingest_parallel > 1 and sample_arg is None:
            parallel_ingest(args)
        else:
            ingest_cmd = [
                PYTHON_EXEC,
                "benchmark/locomo/import_to_neuramem.py",
                "--input", args.data,
            ]
            if sample_arg is not None:
                ingest_cmd.extend(["--sample", str(sample_arg)])
            if args.max_sessions is not None:
                ingest_cmd.extend(["--max-sessions", str(args.max_sessions)])

            if not run_cmd(ingest_cmd):
                logger.error("Ingest failed!")
                sys.exit(1)

    # 2. Run QA Eval & Inline Judge
    logger.info("=== STEP 2: Running Memory Retrieval & QA Evaluation + Judge ===")
    eval_cmd = [
        PYTHON_EXEC,
        "benchmark/locomo/run_eval.py",
        "--input", args.data,
        "--output", args.output,
        "--threads", str(args.threads),
    ]
    if sample_arg is not None:
        eval_cmd.extend(["--sample", str(sample_arg)])
    if args.qa_count is not None:
        eval_cmd.extend(["--count", str(args.qa_count)])

    if not run_cmd(eval_cmd):
        logger.error("QA evaluation failed!")
        sys.exit(1)

    # 3. Calculate Statistics
    logger.info("=== STEP 3: Aggregating Benchmark Metrics & Accuracy Scorecard ===")
    stat_cmd = [
        PYTHON_EXEC,
        "benchmark/locomo/stat_results.py",
        "--input", args.output,
    ]
    run_cmd(stat_cmd)


if __name__ == "__main__":
    main()
