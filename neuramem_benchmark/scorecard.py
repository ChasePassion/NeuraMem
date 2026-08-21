"""Build a scorecard from per-sample LoCoMo run artifacts.

The resumable Windows workflow stores one directory per sample:
``result/locomo_full_rerun/sample_00`` through ``sample_09``.  This module
is the single reader for those directories.  It reconciles the manifest,
worker status, and report files and can optionally verify trace ids against
the live Milvus collection.

Examples::

    python -m neuramem_benchmark.scorecard \
        --root result/locomo_full_rerun \
        --output result/locomo_full_rerun/scorecard.md

    python -m neuramem_benchmark.scorecard \
        --root result/locomo_full_rerun \
        --verify-ghosts \
        --json-output result/locomo_full_rerun/scorecard.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_INT_FIELDS = {
    "Graded Questions": "graded_questions",
    "Correct": "correct",
    "Wrong": "wrong",
}


@dataclass(frozen=True)
class SampleScore:
    """Normalized result metadata for one LoCoMo sample."""

    sample_index: int
    status: str
    status_source: str
    sessions: int | None
    memories_added: int | None
    store_count: int | None
    failed_turns: int | None
    graded_questions: int | None
    correct: int | None
    wrong: int | None
    accuracy: float | None
    average_latency_seconds: float | None

    @property
    def complete(self) -> bool:
        """Whether ingest and evaluation artifacts are present."""
        return self.status == "completed"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def _match_value(text: str, label: str, pattern: str) -> str | None:
    match = re.search(
        rf"(?m)^{re.escape(label)}\s*:\s*{pattern}\s*$",
        text,
    )
    return match.group(1) if match else None


def _match_int(text: str, label: str) -> int | None:
    value = _match_value(text, label, r"(\d+)")
    return int(value) if value is not None else None


def _match_float(text: str, label: str, suffix: str = "") -> float | None:
    value = _match_value(text, label, rf"([0-9]+(?:\.[0-9]+)?){suffix}")
    return float(value) if value is not None else None


def _has_complete_artifacts(sample_dir: Path, sample_index: int) -> bool:
    required = (
        sample_dir / f"ingest_manifest_{sample_index}.json",
        sample_dir / "eval.csv",
        sample_dir / "summary.txt",
    )
    return all(path.exists() for path in required)


def load_sample_score(root: str | Path, sample_index: int) -> SampleScore:
    """Load one sample, inferring completion from artifacts when needed.

    A worker status file is authoritative when it says ``completed``.  If a
    manually launched worker did not write that file, a manifest, eval CSV,
    and summary are sufficient to classify the run as completed.  This keeps
    a valid run from being hidden by an orchestration bookkeeping gap.
    """

    sample_dir = Path(root) / f"sample_{sample_index:02d}"
    manifest = _read_json(sample_dir / f"ingest_manifest_{sample_index}.json")
    worker_status = _read_json(sample_dir / "worker_status.json")
    summary_path = sample_dir / "summary.txt"
    summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""

    artifact_complete = _has_complete_artifacts(sample_dir, sample_index)
    if worker_status and worker_status.get("status") == "completed":
        status = "completed"
        status_source = "worker_status"
    elif artifact_complete:
        status = "completed"
        status_source = "artifacts"
    elif worker_status and worker_status.get("status"):
        status = str(worker_status["status"])
        status_source = "worker_status"
    else:
        status = "incomplete"
        status_source = "artifacts"

    accuracy = _match_float(summary, "Overall Accuracy", "%")
    if accuracy is not None:
        accuracy /= 100.0

    values = {
        field: _match_int(summary, label)
        for label, field in _INT_FIELDS.items()
    }
    return SampleScore(
        sample_index=sample_index,
        status=status,
        status_source=status_source,
        sessions=manifest.get("sessions") if manifest else None,
        memories_added=manifest.get("memories_added") if manifest else None,
        store_count=manifest.get("store_count") if manifest else None,
        failed_turns=manifest.get("failed_turns") if manifest else None,
        graded_questions=values["graded_questions"],
        correct=values["correct"],
        wrong=values["wrong"],
        accuracy=accuracy,
        average_latency_seconds=_match_float(
            summary, "Average Latency per Query", "s"
        ),
    )


def discover_samples(root: str | Path) -> list[int]:
    """Return sample indices represented by ``sample_NN`` directories."""

    pattern = re.compile(r"^sample_(\d+)$")
    indices = []
    for path in Path(root).glob("sample_*"):
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(set(indices))


def load_scorecard(
    root: str | Path,
    sample_indices: Iterable[int] | None = None,
) -> list[SampleScore]:
    """Load scores for selected samples or every sample directory."""

    indices = (
        sorted(set(sample_indices))
        if sample_indices is not None
        else discover_samples(root)
    )
    return [load_sample_score(root, index) for index in indices]


def aggregate_scores(scores: Iterable[SampleScore]) -> dict[str, Any]:
    """Aggregate completed sample metrics using question-weighted accuracy."""

    completed = [
        score
        for score in scores
        if score.complete and score.graded_questions is not None
    ]
    graded = sum(score.graded_questions or 0 for score in completed)
    correct = sum(score.correct or 0 for score in completed)
    wrong = sum(score.wrong or 0 for score in completed)
    failed_turns = sum(score.failed_turns or 0 for score in completed)
    sessions = sum(score.sessions or 0 for score in completed)
    memories_added = sum(score.memories_added or 0 for score in completed)
    weighted_latency_total = sum(
        (score.average_latency_seconds or 0.0) * (score.graded_questions or 0)
        for score in completed
    )
    return {
        "samples": len(completed),
        "graded_questions": graded,
        "correct": correct,
        "wrong": wrong,
        "accuracy": correct / graded if graded else None,
        "average_latency_seconds": (
            weighted_latency_total / graded if graded else None
        ),
        "sessions": sessions,
        "memories_added": memories_added,
        "failed_turns": failed_turns,
    }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def _number(value: int | float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def render_markdown(
    root: str | Path,
    scores: list[SampleScore],
    ghost_checks: dict[int, dict[str, int]] | None = None,
) -> str:
    """Render a human-readable scorecard."""

    aggregate = aggregate_scores(scores)
    lines = [
        "# LoCoMo Benchmark Scorecard",
        "",
        f"- Result root: `{root}`",
        "- Category 5 adversarial questions are excluded, matching `report.py`.",
        "- Accuracy is weighted by graded question count.",
        "",
        "## Overall",
        "",
        "| Samples | Graded | Correct | Wrong | Accuracy | Weighted latency | Failed turns |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {aggregate['samples']} | {aggregate['graded_questions']} | "
            f"{aggregate['correct']} | {aggregate['wrong']} | "
            f"{_percent(aggregate['accuracy'])} | "
            f"{_number(aggregate['average_latency_seconds'])}s | "
            f"{aggregate['failed_turns']} |"
        ),
        "",
        "## Per Sample",
        "",
    ]
    columns = [
        "Sample",
        "Status",
        "Sessions",
        "Memories",
        "Store rows",
        "Failed turns",
        "Graded",
        "Correct",
        "Accuracy",
        "Latency",
    ]
    if ghost_checks is not None:
        columns.append("Ghost IDs")
    lines.extend(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---:" if index else "---" for index in range(len(columns))) + " |",
        ]
    )
    for score in scores:
        status = score.status
        if score.status_source == "artifacts" and status == "completed":
            status += "*"
        row = [
            f"s{score.sample_index}",
            status,
            _number(score.sessions),
            _number(score.memories_added),
            _number(score.store_count),
            _number(score.failed_turns),
            _number(score.graded_questions),
            _number(score.correct),
            _percent(score.accuracy),
            f"{_number(score.average_latency_seconds)}s",
        ]
        if ghost_checks is not None:
            row.append(_number(ghost_checks.get(score.sample_index, {}).get("ghosts")))
        lines.append("| " + " | ".join(row) + " |")
    if any(score.status_source == "artifacts" for score in scores):
        lines.extend(
            [
                "",
                "`*` means completion was inferred from the manifest, eval CSV, and summary because the worker status file was absent or stale.",
            ]
        )
    if ghost_checks is not None:
        lines.extend(
            [
                "",
                "Ghost IDs are episodic IDs present in a trace but absent from the current Milvus store.",
            ]
        )
    return "\n".join(lines) + "\n"


def verify_ghost_ids(
    root: str | Path,
    scores: Iterable[SampleScore],
    milvus_uri: str,
    collection_name: str = "memories",
) -> dict[int, dict[str, int]]:
    """Compare trace episodic IDs with the current Milvus user rows."""

    from pymilvus import MilvusClient

    client = MilvusClient(uri=milvus_uri)
    checks: dict[int, dict[str, int]] = {}
    for score in scores:
        sample_dir = Path(root) / f"sample_{score.sample_index:02d}"
        trace_path = sample_dir / "eval.trace.jsonl"
        if not trace_path.exists():
            checks[score.sample_index] = {"store": 0, "trace_ids": 0, "ghosts": 0}
            continue
        rows = client.query(
            collection_name=collection_name,
            filter=(
                f'user_id == "sample_{score.sample_index}" '
                'and memory_type == "episodic"'
            ),
            output_fields=["id"],
            limit=16384,
        )
        store_ids = {row["id"] for row in rows}
        trace_ids: set[int] = set()
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for memory in payload.get("retrieval", {}).get("episodic", []):
                    if "id" in memory:
                        trace_ids.add(int(memory["id"]))
        checks[score.sample_index] = {
            "store": len(store_ids),
            "trace_ids": len(trace_ids),
            "ghosts": len(trace_ids - store_ids),
        }
    return checks


def _parse_samples(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a LoCoMo benchmark scorecard")
    parser.add_argument("--root", default="result/locomo_full_rerun")
    parser.add_argument("--samples", help="Comma-separated sample indices")
    parser.add_argument("--output", help="Write Markdown scorecard to this path")
    parser.add_argument("--json-output", help="Write machine-readable scorecard JSON")
    parser.add_argument(
        "--verify-ghosts",
        action="store_true",
        help="Compare trace episodic IDs with the current Milvus store",
    )
    parser.add_argument("--milvus-uri", default=None)
    args = parser.parse_args()

    scores = load_scorecard(args.root, _parse_samples(args.samples))
    ghost_checks = None
    if args.verify_ghosts:
        milvus_uri = args.milvus_uri or os.getenv("MILVUS_URL")
        if not milvus_uri:
            raise SystemExit("--verify-ghosts requires --milvus-uri or MILVUS_URL")
        ghost_checks = verify_ghost_ids(args.root, scores, milvus_uri)

    markdown = render_markdown(args.root, scores, ghost_checks)
    print(markdown, end="")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(args.root),
            "aggregate": aggregate_scores(scores),
            "samples": [
                {
                    **asdict(score),
                    "ghost_check": (
                        ghost_checks.get(score.sample_index)
                        if ghost_checks is not None
                        else None
                    ),
                }
                for score in scores
            ],
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
