"""LoCoMo ingest replay (async port of import_to_neuramem.py).

Behavior preserved from the legacy importer: paired-turn replay with
date prefixes, per-turn manage error isolation (#22), consolidation
every 7 sessions (never after the last), per-sample usage JSON for the
merged report.

NEW — ingest completeness manifests (s8 lesson, RUN_RECORD 8.1): each
sample writes ingest_manifest_{idx}.json with the final store count.
The eval runner refuses samples without a manifest, so a truncated
ingest can never silently produce a fake score again.
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import dotenv

from neuramem.config import MemoryConfig
from neuramem.core.models import MemoryFilter
from neuramem.llm.openai_adapter import OpenAILLM
from neuramem.memory import Memory
from neuramem_benchmark.llm_config import build_benchmark_config
from neuramem_benchmark.locomo import load_locomo_samples

dotenv.load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONSOLIDATE_EVERY_SESSIONS = 7


def _format_msg(msg: Any) -> str:
    if isinstance(msg, dict):
        speaker = msg.get("speaker", "Speaker")
        text = msg.get("text", "")
        return f"{speaker}: {text}"
    return str(msg)


async def _run_consolidation(memory: Memory, user_id: str, sample_name: str) -> None:
    """Best-effort consolidation; failures must not abort the ingest."""
    try:
        stats = await memory.consolidate_async(user_id)
        logger.info(
            "Consolidated %s: processed=%d, semantic_created=%d",
            sample_name, stats.memories_processed, stats.semantic_created,
        )
    except Exception as e:  # noqa: BLE001 - consolidation is best-effort
        logger.warning("Consolidation failed for %s: %s", sample_name, e)


async def import_sample(
    memory: Memory,
    sample: Dict[str, Any],
    reset_first: bool = True,
    max_sessions: Optional[int] = None,
) -> Dict[str, Any]:
    """Replay one conversation sample; returns manifest data."""
    user_id = sample["user_id"]
    sample_name = sample.get("sample_id", f"sample_{sample['sample_index']}")

    if reset_first:
        await memory.reset_async(user_id)
        logger.info("Reset memory for user %s (%s)", user_id, sample_name)

    conv = sample.get("conversation", {})
    session_keys = []
    for k in conv.keys():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv[k], list):
            try:
                session_keys.append((int(k.split("_")[1]), k))
            except ValueError:
                pass
    session_keys.sort(key=lambda x: x[0])
    selected_keys = [k for _, k in session_keys]
    if max_sessions is not None:
        selected_keys = selected_keys[:max_sessions]

    total_added = 0
    failed_turns = 0
    processed_sessions = 0
    pbar = tqdm(selected_keys, desc=f"Ingesting {sample_name}")
    for s_key in pbar:
        messages = conv[s_key]
        if not messages or not isinstance(messages, list):
            continue
        date_time = conv.get(f"{s_key}_date_time", "")
        prefix = f"[{date_time}] " if date_time else ""

        i = 0
        while i < len(messages):
            user_text = prefix + _format_msg(messages[i])
            if i + 1 < len(messages):
                assistant_text = _format_msg(messages[i + 1])
                i += 2
            else:
                assistant_text = "(No reply)"
                i += 1
            try:
                added_ids = await memory.manage_async(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    user_id=user_id,
                    chat_id=s_key,
                )
            except Exception as e:  # noqa: BLE001 - per-turn isolation (#22)
                failed_turns += 1
                logger.warning(
                    "manage failed for turn in %s/%s, skipping turn "
                    "(%d failed so far): %s",
                    sample_name, s_key, failed_turns, e,
                )
                continue
            total_added += len(added_ids)
            pbar.set_postfix({"added_memories": total_added})

        processed_sessions += 1
        if processed_sessions % CONSOLIDATE_EVERY_SESSIONS == 0:
            await _run_consolidation(memory, user_id, sample_name)

    final_count = await memory.store.count(MemoryFilter(user_id=user_id))
    logger.info(
        "Successfully ingested %s: %d episodic memories stored, store count=%d "
        "(%d turns skipped on manage failure).",
        sample_name, total_added, final_count, failed_turns,
    )
    return {
        "sample_index": sample["sample_index"],
        "sample_id": sample_name,
        "user_id": user_id,
        "memories_added": total_added,
        "store_count": final_count,
        "sessions": processed_sessions,
        "failed_turns": failed_turns,
    }


def write_manifest(manifest: Dict[str, Any], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"ingest_manifest_{manifest['sample_index']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Ingest manifest written to %s (count=%d)",
                path, manifest["store_count"])
    return path


def write_ingest_usage(llm: OpenAILLM, output_dir: str, sample_idx: Optional[int]) -> None:
    """Persist ingest-phase memory-system usage (merged by report.py)."""
    stats = llm.usage_stats
    payload = {
        "sample_index": sample_idx,
        "manage": stats.snapshot("manage"),
        "consolidate": stats.snapshot("consolidate"),
        "total": stats.snapshot(),
    }
    os.makedirs(output_dir, exist_ok=True)
    name = f"ingest_usage_stats_{sample_idx}.json" if sample_idx is not None else "ingest_usage_stats.json"
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Ingest usage stats written to %s", path)


async def run(args) -> None:
    config = build_benchmark_config(args.milvus_uri)
    logger.info("Initializing NeuraMem with Milvus URI: %s", config.store.uri)
    # one shared LLM instance: single usage aggregate (W3 topology),
    # injected instead of reaching into the facade
    llm = OpenAILLM(config.llm)
    memory = Memory(config, llm=llm)

    samples: List[Dict[str, Any]] = load_locomo_samples(args.input, args.sample)
    logger.info("Loaded %d sample(s) from %s", len(samples), args.input)

    start_time = time.time()
    for sample in samples:
        manifest = await import_sample(
            memory, sample,
            reset_first=not args.no_reset,
            max_sessions=args.max_sessions,
        )
        write_manifest(manifest, args.usage_output_dir)

    elapsed = time.time() - start_time
    logger.info("Import complete! in %.1fs", elapsed)
    write_ingest_usage(llm, args.usage_output_dir, args.sample)


def main():
    parser = argparse.ArgumentParser(description="Import LoCoMo conversations into NeuraMem")
    parser.add_argument("--input", default="data/locomo10.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--milvus-uri", default=None)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--usage-output-dir", default="result")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
