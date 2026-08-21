"""App-side snowflake id generation.

The Milvus collections use ``auto_id=False`` so that ``upsert`` keeps the
primary key stable (#17). Fresh ids are assigned at insert time by the
store from this generator — Milvus auto-id would silently re-id every
upserted row (delete + insert-with-new-id), orphaning every id a caller
ever saw.

Layout: 41-bit millisecond timestamp (custom epoch) | 10-bit worker id
(derived from the OS pid, unique across concurrent local processes) |
12-bit per-ms sequence — unique and time-ordered, well under int64.
"""

import os
import threading
import time

# 2025-02-20T02:13:20Z. Keeps current ids near 2e17: below the old
# Milvus auto-id range (~4.7e17) carried over by migration, so fresh and
# migrated ids cannot collide.
_EPOCH_MS = 1_740_000_000_000

_lock = threading.Lock()
_last_ts = 0
_seq = 0
_worker_id = os.getpid() & 0x3FF


def new_id() -> int:
    """Return the next snowflake id (unique within this host)."""
    global _last_ts, _seq
    with _lock:
        ts = int(time.time() * 1000) - _EPOCH_MS
        if ts < _last_ts:
            # clock stepped backwards — reuse the last bucket and burn
            # sequence slots instead of risking a duplicate
            ts = _last_ts
        if ts == _last_ts:
            _seq = (_seq + 1) & 0xFFF
            if _seq == 0:
                while ts == _last_ts:
                    ts = int(time.time() * 1000) - _EPOCH_MS
        else:
            _seq = 0
        _last_ts = ts
        return (ts << 22) | (_worker_id << 12) | _seq
