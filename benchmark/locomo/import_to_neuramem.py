"""LoCoMo dataset importer for NeuraMem.

Reads LoCoMo JSON conversations and ingests dialogue turns into NeuraMem.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dotenv
dotenv.load_dotenv(PROJECT_ROOT / ".env")

from src.memory_system import Memory, MemoryConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_locomo_samples(path: str, sample_idx: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load LoCoMo JSON dataset."""
    if not os.path.exists(path):
        alt_path = Path("E:/code/locomo/data/locomo10.json")
        if alt_path.exists():
            path = str(alt_path)
        else:
            raise FileNotFoundError(f"LoCoMo dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for idx, item in enumerate(data):
        normalized = dict(item)
        normalized["sample_index"] = idx
        normalized["user_id"] = f"sample_{idx}"
        samples.append(normalized)

    if sample_idx is not None:
        return [s for s in samples if s["sample_index"] == sample_idx]
    return samples


def _format_msg(msg: Any) -> str:
    if isinstance(msg, dict):
        speaker = msg.get("speaker", "Speaker")
        text = msg.get("text", "")
        return f"{speaker}: {text}"
    return str(msg)


def import_sample(
    memory: Memory,
    sample: Dict[str, Any],
    reset_first: bool = True,
    max_sessions: Optional[int] = None,
) -> int:
    """Import a single conversation sample into NeuraMem."""
    user_id = sample["user_id"]
    sample_name = sample.get("sample_id", f"sample_{sample['sample_index']}")
    conv = sample.get("conversation", {})

    if reset_first:
        memory.reset(user_id)
        logger.info(f"Reset memory for user {user_id} ({sample_name})")

    # Filter only actual session list keys (session_1, session_2, ...)
    session_keys = []
    for k in conv.keys():
        if k.startswith("session_") and not k.endswith("_date_time") and isinstance(conv[k], list):
            try:
                num = int(k.split("_")[1])
                session_keys.append((num, k))
            except ValueError:
                pass

    session_keys.sort(key=lambda x: x[0])
    selected_keys = [k for _, k in session_keys]
    if max_sessions is not None:
        selected_keys = selected_keys[:max_sessions]

    total_added = 0
    pbar = tqdm(selected_keys, desc=f"Ingesting {sample_name}")
    for s_key in pbar:
        messages = conv[s_key]
        if not messages or not isinstance(messages, list):
            continue

        date_time = conv.get(f"{s_key}_date_time", "")
        prefix = f"[{date_time}] " if date_time else ""

        # Ingest dialogue turns in pairs
        i = 0
        while i < len(messages):
            msg1 = messages[i]
            user_text = prefix + _format_msg(msg1)

            if i + 1 < len(messages):
                msg2 = messages[i + 1]
                assistant_text = _format_msg(msg2)
                i += 2
            else:
                assistant_text = "(No reply)"
                i += 1

            added_ids = memory.manage(
                user_text=user_text,
                assistant_text=assistant_text,
                user_id=user_id,
                chat_id=s_key,
            )
            total_added += len(added_ids)
            pbar.set_postfix({"added_memories": total_added})

    logger.info(f"Successfully ingested {sample_name}: {total_added} episodic memories stored.")
    return total_added


def main():
    parser = argparse.ArgumentParser(description="Import LoCoMo conversations into NeuraMem")
    parser.add_argument("--input", default="data/locomo10.json", help="Path to locomo10.json")
    parser.add_argument("--sample", type=int, default=None, help="Sample index (0-9) to import. None = all.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Max sessions per sample")
    parser.add_argument("--milvus-uri", default=None, help="Milvus connection URI (defaults to .env or neuramem_bench.db)")
    parser.add_argument("--no-reset", action="store_true", help="Do not reset existing user memories before ingest")
    args = parser.parse_args()

    config = MemoryConfig()
    if args.milvus_uri:
        config.milvus_uri = args.milvus_uri
    elif not config.milvus_uri:
        config.milvus_uri = os.getenv("MILVUS_URL", "http://117.72.161.187:19530")

    logger.info(f"Initializing NeuraMem with Milvus URI: {config.milvus_uri}")
    memory = Memory(config)

    samples = load_locomo_samples(args.input, args.sample)
    logger.info(f"Loaded {len(samples)} sample(s) from {args.input}")

    total_all_memories = 0
    start_time = time.time()
    for sample in samples:
        count = import_sample(
            memory=memory,
            sample=sample,
            reset_first=not args.no_reset,
            max_sessions=args.max_sessions,
        )
        total_all_memories += count

    elapsed = time.time() - start_time
    logger.info(f"Import complete! Total memories added: {total_all_memories} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
