"""Statistics aggregator for LoCoMo benchmark evaluation results.

Computes accuracy per category (multi-hop, temporal, open-domain, single-hop),
excluding Category 5 adversarial questions, exactly like OpenViking's benchmark reporting.
"""

import argparse
import csv
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


def main():
    parser = argparse.ArgumentParser(description="Statistics for judge result CSV")
    parser.add_argument(
        "--input",
        default="result/locomo_neuramem_results.csv",
        help="Path to judge result CSV file",
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

    total_graded = correct + wrong
    accuracy = correct / total_graded if total_graded > 0 else 0.0
    avg_time = total_time / valid_rows if valid_rows > 0 else 0.0
    cache_rate = (
        cache_hit_tokens / cache_prompt_tokens
        if cache_prompt_tokens > 0 else None
    )

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
        f"Cache Hit Rate (tokens)       : {cache_rate:.2%}" if cache_rate is not None
        else "Cache Hit Rate (tokens)       : n/a (no cache tokens reported)",
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
