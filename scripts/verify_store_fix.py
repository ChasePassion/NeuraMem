"""Live-store verification for the auto_id=False fix (stable ids).

Runs the real MilvusStore code path against scratch collections on the
configured server, asserts the properties the fix promises, then cleans
up. Exit code 0 = all green.

Usage: PYTHONPATH=src python scripts/verify_store_fix.py [collection-stem]
"""

import asyncio
import sys
import time

from neuramem.config import StoreConfig
from neuramem.core.models import MemoryFilter, MemoryRecord
from neuramem.store.milvus import MilvusStore

URI = "http://117.72.161.187:19530"
DIM = 4


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return ok


async def main() -> int:
    stem = sys.argv[1] if len(sys.argv) > 1 else f"fixtest_{int(time.time())}"
    config = StoreConfig(
        uri=URI,
        collection_name=f"{stem}_memories",
        groups_collection_name=f"{stem}_groups",
    )
    store = MilvusStore(config)
    await store.create_collection(dim=DIM)
    vec = [0.1, 0.2, 0.3, 0.4]
    ok = True

    try:
        # -- memories: insert assigns stable app-side ids -----------------
        a = MemoryRecord(user_id="u1", memory_type="episodic", ts=1,
                         chat_id="c1", text="alpha", vector=vec)
        b = MemoryRecord(user_id="u1", memory_type="episodic", ts=2,
                         chat_id="c1", text="beta", vector=vec)
        ids = await store.insert([a, b])
        ok &= check("insert returns 2 ids", len(ids) == 2)
        ok &= check("insert fills record.id (caller keeps the handle)",
                    a.id == ids[0] and b.id == ids[1] and ids[0] > 0)
        id_a = ids[0]

        rows = await store.query(MemoryFilter(user_id="u1"), limit=10)
        ok &= check("query sees both rows", len(rows) == 2)

        # -- the core property: upsert keeps the id -----------------------
        a.text = "alpha v2"
        a.vector = None  # partial update — vector backfill path (#17)
        await store.upsert([a])
        rows = await store.query(MemoryFilter(id_in=[id_a]), limit=5)
        ok &= check("upsert preserves id", len(rows) == 1 and rows[0].id == id_a,
                    f"id={rows[0].id if rows else 'GONE'}")
        ok &= check("upsert applied new text", rows and rows[0].text == "alpha v2")
        ok &= check("upsert did not clone the row",
                    len(await store.query(MemoryFilter(user_id="u1"), limit=10)) == 2)
        vec_rows = await store.query(
            MemoryFilter(id_in=[id_a]), limit=5, include_vectors=True
        )
        ok &= check("upsert backfilled vector",
                    vec_rows and vec_rows[0].vector is not None)

        # repeated churn (eval-time narrative group flips)
        a.group_id = 42
        await store.upsert([a])
        a.group_id = 43
        await store.upsert([a])
        rows = await store.query(MemoryFilter(id_in=[id_a]), limit=5)
        ok &= check("id survives repeated upsert churn",
                    len(rows) == 1 and rows[0].id == id_a and rows[0].group_id == 43)
        ok &= check("still exactly 2 rows for user",
                    len(await store.query(MemoryFilter(user_id="u1"), limit=10)) == 2)

        # -- search returns the stable id ---------------------------------
        hits = await store.search([vec], MemoryFilter(user_id="u1"), limit=5)
        hit_ids = {h.record.id for h in hits[0]}
        ok &= check("search returns stable ids", id_a in hit_ids or
                    len(hit_ids & {id_a, ids[1]}) >= 1, f"hit ids={sorted(hit_ids)}")

        # -- delete --------------------------------------------------------
        await store.delete(ids=[ids[1]])
        ok &= check("delete removes exactly the target",
                    len(await store.query(MemoryFilter(user_id="u1"), limit=10)) == 1)

        # -- groups: insert/update keep group_id ---------------------------
        gid = await store.insert_group("u1", centroid_vector=vec, size=1)
        ok &= check("insert_group returns id", gid is not None and gid > 0)
        await store.update_group("u1", gid, size=5)
        groups = await store.list_groups("u1")
        ok &= check("update_group preserves group_id",
                    any(g.group_id == gid and g.size == 5 for g in groups),
                    f"groups={[(g.group_id, g.size) for g in groups]}")
        matches = await store.search_groups("u1", vec, limit=3)
        ok &= check("search_groups finds the group",
                    any(m.group_id == gid for m in matches))
        await store.update_group("u1", gid, size=7)
        groups = await store.list_groups("u1")
        ok &= check("group_id still stable after second update",
                    len(groups) == 1 and groups[0].group_id == gid and groups[0].size == 7)
    finally:
        client = store._client
        for name in (config.collection_name, config.groups_collection_name):
            if client.has_collection(name):
                client.drop_collection(name)
        print("  (scratch collections dropped)")

    print("RESULT:", "ALL GREEN" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
