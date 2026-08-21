"""Cross-platform one-shot LoCoMo pipeline: ingest, evaluate, and report.

Use ``scripts/locomo_batch.ps1`` when a Windows run needs resumable,
per-sample process supervision instead of one combined output file.
"""

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from neuramem_benchmark.locomo import count_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PYTHON_EXEC = os.getenv("BENCHMARK_PYTHON") or sys.executable


def run_cmd(cmd_list: list[str]) -> bool:
    logger.info("Running: %s", " ".join(cmd_list))
    res = subprocess.run(cmd_list, shell=False)
    return res.returncode == 0


def ingest_cmd(data: str, sample: int | None, max_sessions: int | None) -> list[str]:
    cmd = [PYTHON_EXEC, "-m", "neuramem_benchmark.ingest", "--input", data]
    if sample is not None:
        cmd.extend(["--sample", str(sample)])
    if max_sessions is not None:
        cmd.extend(["--max-sessions", str(max_sessions)])
    return cmd


def parallel_ingest(args) -> None:
    """One subprocess per sample, up to --ingest-parallel at once."""
    n_samples = count_samples(args.data)
    if n_samples <= 1:
        if not run_cmd(ingest_cmd(args.data, None, args.max_sessions)):
            logger.error("Ingest failed!")
            sys.exit(1)
        return

    workers = min(args.ingest_parallel, n_samples)
    logger.info("Parallel ingest: %d worker(s) across %d samples", workers, n_samples)

    def ingest_one(sample_idx: int) -> bool:
        return run_cmd(ingest_cmd(args.data, sample_idx, args.max_sessions))

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(ingest_one, i): i for i in range(n_samples)}
        for f in as_completed(futures):
            sample_idx = futures[f]
            try:
                ok = f.result()
            except Exception as e:  # noqa: BLE001 - report and continue
                ok = False
                logger.error("Sample %d ingest raised: %s", sample_idx, e)
            if not ok:
                failed.append(sample_idx)
                logger.error("Sample %d ingest failed", sample_idx)
    if failed:
        logger.error("Ingest failed for samples: %s", sorted(failed))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="End-to-end LoCoMo Benchmark for NeuraMem")
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--output", default="result/locomo_neuramem_results.csv")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--all-samples", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--qa-count", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ingest-parallel", type=int, default=4)
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    sample_arg = None if args.all_samples else args.sample

    print("=" * 80)
    print("      Starting NeuraMem LoCoMo Benchmark     ")
    print(f"      Dataset: {args.data} | Sample: {'ALL' if args.all_samples else sample_arg}    ")
    print("=" * 80)

    if not args.skip_ingest:
        logger.info("=== STEP 1: Ingesting Conversation Dialogue into NeuraMem ===")
        if args.ingest_parallel > 1 and sample_arg is None:
            parallel_ingest(args)
        elif not run_cmd(ingest_cmd(args.data, sample_arg, args.max_sessions)):
            logger.error("Ingest failed!")
            sys.exit(1)

    logger.info("=== STEP 2: Running Memory Retrieval & QA Evaluation + Judge ===")
    eval_cmd = [
        PYTHON_EXEC, "-m", "neuramem_benchmark.runner",
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

    logger.info("=== STEP 3: Aggregating Benchmark Metrics & Accuracy Scorecard ===")
    run_cmd([PYTHON_EXEC, "-m", "neuramem_benchmark.report", "--input", args.output])


if __name__ == "__main__":
    main()
