"""Statistics aggregator for LoCoMo benchmark evaluation results.

Computes accuracy per category (multi-hop, temporal, open-domain, single-hop),
excluding Category 5 adversarial questions, exactly like OpenViking's benchmark
reporting. Also merges KV cache (prefix cache) usage across the whole memory
system: ingest phase (manage/consolidate, from ingest_usage_stats*.json) and
eval phase (usage_judge + answer, from the QA CSV). Judge calls are an
evaluation tool and are excluded from the memory-system rate.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}

csv.field_size_limit(sys.maxsize)


def category_label(category: str) -> str:
    category = str(category or "").strip()
    name = CATEGORY_NAMES.get(category)
    if name:
        return f"{category}-{name}"
    return category or "<missing>"


def _prompt_of(snapshot: dict) -> float:
    return (
        snapshot["input_tokens"]
        + snapshot["cache_read_tokens"]
        + snapshot["cache_write_tokens"]
    )


def _merge(*snapshots: dict) -> dict:
    merged = {k: 0 for k in ("calls", "input_tokens", "output_tokens",
                             "cache_read_tokens", "cache_write_tokens",
                             "reasoning_tokens", "total_tokens", "cost")}
    for s in snapshots:
        if not s:
            continue
        for k in merged:
            merged[k] += s.get(k, 0)
    return merged


def _rate(prompt: float, hit: float) -> float | None:
    return hit / prompt if prompt > 0 else None


def load_ingest_usage(usage_dir: str) -> tuple[dict, dict, list[str]]:
    """Load ingest-phase memory-system usage from usage JSON files.

    Reads result/ingest_usage_stats*.json written by import_to_neuramem.py
    (serial run writes one file, parallel sample subprocesses write one per
    sample). Returns (merged manage usage, merged consolidate usage, files).
    """
    manage_usage: dict = {}
    consolidate_usage: dict = {}
    files = sorted(glob.glob(os.path.join(usage_dir, "ingest_usage_stats*.json")))
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError):
            continue
        manage_usage = _merge(manage_usage, payload.get("manage") or {})
        consolidate_usage = _merge(consolidate_usage, payload.get("consolidate") or {})
    return manage_usage, consolidate_usage, files


def main():
    parser = argparse.ArgumentParser(description="Statistics for judge result CSV")
    parser.add_argument(
        "--input",
        default="result/locomo_neuramem_results.csv",
        help="Path to judge result CSV file",
    )
    parser.add_argument(
        "--ingest-usage-dir",
        default="result",
        help=(
            "Directory containing ingest_usage_stats*.json from "
            "import_to_neuramem.py; merged into the memory-system rate"
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    correct = 0
    wrong = 0
    total_time = 0.0
    valid_rows = 0
    by_category = defaultdict(lambda: {"CORRECT": 0, "WRONG": 0, "OTHER": 0})
    cache_hit_tokens = 0
    cache_prompt_tokens = 0
    answer_cache_hit_tokens = 0
    answer_cache_prompt_tokens = 0
    memory_cache_hit_tokens = 0
    memory_cache_prompt_tokens = 0
    cache_rows = 0

    with open(args.input, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = str(row.get("category", "")).strip()
            # Skip adversarial category 5
            if category == "5":
                continue

            valid_rows += 1
            cat_key = category_label(category)
            res = str(row.get("result", "")).strip().upper()

            if "CORRECT" in res:
                correct += 1
                by_category[cat_key]["CORRECT"] += 1
            elif "WRONG" in res:
                wrong += 1
                by_category[cat_key]["WRONG"] += 1
            else:
                by_category[cat_key]["OTHER"] += 1

            time_cost = row.get("time_cost", "")
            if time_cost:
                try:
                    total_time += float(time_cost)
                except ValueError:
                    pass

            # KV/prefix cache usage reported per question (fields absent in
            # CSVs produced before cache accounting was added -> treated as 0)
            hit = row.get("cache_hit_tokens", "")
            prompt = row.get("cache_prompt_tokens", "")
            if hit:
                try:
                    cache_hit_tokens += float(hit)
                    cache_prompt_tokens += float(prompt)
                    cache_rows += 1
                except ValueError:
                    pass
            answer_hit = row.get("answer_cache_hit_tokens", "")
            if answer_hit:
                try:
                    answer_cache_hit_tokens += float(answer_hit)
                    answer_cache_prompt_tokens += float(
                        row.get("answer_cache_prompt_tokens", "")
                    )
                except ValueError:
                    pass
            memory_hit = row.get("memory_cache_hit_tokens", "")
            if memory_hit:
                try:
                    memory_cache_hit_tokens += float(memory_hit)
                    memory_cache_prompt_tokens += float(
                        row.get("memory_cache_prompt_tokens", "")
                    )
                except ValueError:
                    pass

    total_graded = correct + wrong
    accuracy = correct / total_graded if total_graded > 0 else 0.0
    avg_time = total_time / valid_rows if valid_rows > 0 else 0.0
    cache_rate = _rate(cache_prompt_tokens, cache_hit_tokens)
    # Memory-system components (judge is an eval tool, excluded):
    # - manage / consolidate: ingest phase, from the usage JSON files
    # - usage_judge / answer: eval phase, from the QA CSV (CSV stores the
    #   combined usage_judge+answer slice plus the answer-only slice, so the
    #   usage_judge slice is the difference)
    ingest_usage, consolidate_usage, ingest_files = load_ingest_usage(args.ingest_usage_dir)
    component_manage = _prompt_of(ingest_usage), ingest_usage["cache_read_tokens"]
    component_consolidate = (
        _prompt_of(consolidate_usage),
        consolidate_usage["cache_read_tokens"],
    )
    component_answer = answer_cache_prompt_tokens, answer_cache_hit_tokens
    component_usage_judge = (
        memory_cache_prompt_tokens - answer_cache_prompt_tokens,
        memory_cache_hit_tokens - answer_cache_hit_tokens,
    )
    memory_system_prompt = (
        component_manage[0]
        + component_consolidate[0]
        + component_usage_judge[0]
        + component_answer[0]
    )
    memory_system_hit = (
        component_manage[1]
        + component_consolidate[1]
        + component_usage_judge[1]
        + component_answer[1]
    )
    memory_system_rate = _rate(memory_system_prompt, memory_system_hit)
    # Eval-tool calls (judge): within the eval process only (CSV overall
    # minus the eval-phase memory-system slice; ingest is a separate process)
    aux_prompt = cache_prompt_tokens - memory_cache_prompt_tokens
    aux_rate = _rate(aux_prompt, cache_hit_tokens - memory_cache_hit_tokens)

    def _component_line(name: str, prompt: float, hit: float) -> str:
        rate = _rate(prompt, hit)
        return f"  {name:<16}: " + (f"{rate:.2%}" if rate is not None else "n/a")

    output_lines = [
        "==========================================================================",
        "           NeuraMem LoCoMo Benchmark Evaluation Report                    ",
        "==========================================================================",
        f"Total Questions (excl. cat-5) : {valid_rows}",
        f"Graded Questions              : {total_graded}",
        f"Correct                       : {correct}",
        f"Wrong                         : {wrong}",
        f"Overall Accuracy              : {accuracy:.2%}",
        f"Average Latency per Query     : {avg_time:.2f}s",
        "--------------------------------------------------------------------------",
        "KV Cache (Prefix Cache) Usage:",
        "Memory System Hit Rate        : "
        + (f"{memory_system_rate:.2%}" if memory_system_rate is not None else "n/a")
        + "  (manage + consolidate + usage_judge + answer; judge excluded)",
        _component_line("manage", *component_manage)
        + f"  [{int(ingest_usage['calls'])} calls, "
        + (f"{len(ingest_files)} file(s)]" if ingest_files else "no usage file]"),
        _component_line("consolidate", *component_consolidate)
        + f"  [{int(consolidate_usage['calls'])} calls]",
        _component_line("usage_judge", *component_usage_judge),
        _component_line("answer", *component_answer),
        f"  Memory System Hit / Prompt  : {int(memory_system_hit)} / {int(memory_system_prompt)}",
        "Judge (eval tool, excluded)   : "
        + (f"{aux_rate:.2%}" if aux_rate is not None else "n/a"),
        "Overall Hit Rate (tokens)     : "
        + (f"{cache_rate:.2%}" if cache_rate is not None else "n/a"),
        f"Cache Hit Tokens              : {int(cache_hit_tokens)}",
        f"Cache Prompt Tokens           : {int(cache_prompt_tokens)}",
        f"Questions Reporting Cache     : {cache_rows}/{valid_rows}",
        "--------------------------------------------------------------------------",
        "Breakdown by Category:",
        f"{'Category':<24} {'Correct':>8} {'Wrong':>8} {'Other':>8} {'Total':>8} {'Accuracy':>10}",
        "-" * 74,
    ]

    for cat in sorted(by_category):
        c_correct = by_category[cat]["CORRECT"]
        c_wrong = by_category[cat]["WRONG"]
        c_other = by_category[cat]["OTHER"]
        c_graded = c_correct + c_wrong
        c_total = c_graded + c_other
        c_acc = c_correct / c_graded if c_graded > 0 else 0.0
        output_lines.append(
            f"{cat:<24} {c_correct:>8} {c_wrong:>8} {c_other:>8} {c_total:>8} {c_acc:>9.2%}"
        )

    output_lines.append("==========================================================================")

    for line in output_lines:
        print(line)

    summary_dir = os.path.dirname(args.input) or "."
    summary_path = os.path.join(summary_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nReport saved to: {summary_path}")


if __name__ == "__main__":
    main()
