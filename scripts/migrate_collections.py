"""Migrate production collections to auto_id=False schemas, preserving data.

For each target collection ('memories', 'groups'):
1. export a no-vector JSONL manifest backup (safety net)
2. create <name>_v2 with the fixed schema (same fields/index/consistency)
3. copy every row preserving its primary key
4. verify: row counts, exact id-set equality, random-row full-field
   equality including vectors — abort (drop nothing) on any mismatch
5. drop the old collection and alias the old name to _v2

Idempotent-ish: refuses to run if a previous *_v2 already exists unless
--force-v2 is passed (then it drops the stale _v2 first).
"""

import argparse
import json
import os
import random
import sys
import time

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

URI = "http://117.72.161.187:19530"
BATCH = 500
PAGE = 16384


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def memories_schema(dim: int) -> CollectionSchema:
    fields = [
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("user_id", DataType.VARCHAR, max_length=128),
        FieldSchema("memory_type", DataType.VARCHAR, max_length=32),
        FieldSchema("ts", DataType.INT64),
        FieldSchema("chat_id", DataType.VARCHAR, max_length=128),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema("group_id", DataType.INT64, default_value=-1),
    ]
    return CollectionSchema(fields, enable_dynamic_field=True)


def groups_schema(dim: int) -> CollectionSchema:
    fields = [
        FieldSchema("group_id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("user_id", DataType.VARCHAR, max_length=128),
        FieldSchema("centroid_vector", DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema("size", DataType.INT64),
    ]
    return CollectionSchema(fields, enable_dynamic_field=True)


def fetch_all(client: MilvusClient, name: str, filter_expr: str = "") -> list[dict]:
    rows: list[dict] = []
    it = client.query_iterator(
        collection_name=name, batch_size=PAGE, filter=filter_expr or "",
        output_fields=["*"],
    )
    while True:
        batch = it.next()
        if not batch:
            it.close()
            return rows
        rows.extend(batch)


def vector_dim(client: MilvusClient, name: str, field: str) -> int:
    desc = client.describe_collection(name)
    for f in desc["fields"]:
        if f["name"] == field:
            return f["params"]["dim"]
    raise RuntimeError(f"vector field {field} not found in {name}")


def migrate(client: MilvusClient, name: str, pk: str, vector_field: str,
            metric: str, backup_dir: str) -> bool:
    log(f"=== {name} ===")
    if not client.has_collection(name):
        log(f"  skip: {name} does not exist")
        return True
    v2 = f"{name}_v2"
    if client.has_collection(v2):
        if "--force-v2" not in sys.argv:
            log(f"  ABORT: {v2} already exists (pass --force-v2 to replace)")
            return False
        client.drop_collection(v2)
        log(f"  dropped stale {v2}")

    dim = vector_dim(client, name, vector_field)
    schema = (memories_schema if name == "memories" else groups_schema)(dim)
    idx = client.prepare_index_params()
    idx.add_index(field_name=vector_field, index_type="AUTOINDEX", metric_type=metric)
    client.create_collection(v2, schema=schema, index_params=idx,
                             consistency_level="Strong")
    log(f"  created {v2} (dim={dim}, auto_id=False)")

    rows = fetch_all(client, name)
    log(f"  source rows: {len(rows)}")
    for i in range(0, len(rows), BATCH):
        client.insert(v2, data=rows[i:i + BATCH])
    log(f"  copied {len(rows)} rows preserving pks")

    # -- gate 1: counts ----------------------------------------------------
    new_rows = fetch_all(client, v2)
    if len(new_rows) != len(rows):
        log(f"  ABORT: count mismatch old={len(rows)} new={len(new_rows)}")
        return False
    log(f"  gate1 count OK ({len(new_rows)})")

    # -- gate 2: exact id-set equality --------------------------------------
    old_ids = {r[pk] for r in rows}
    new_ids = {r[pk] for r in new_rows}
    if old_ids != new_ids:
        log(f"  ABORT: id-set mismatch missing={len(old_ids - new_ids)} "
            f"extra={len(new_ids - old_ids)}")
        return False
    log("  gate2 id-set OK")

    # -- gate 3: random-row full-field equality (vectors included) ----------
    by_pk = {r[pk]: r for r in new_rows}
    for r in random.sample(rows, min(20, len(rows))):
        n = by_pk[r[pk]]
        for k, v in r.items():
            nv = n.get(k)
            if isinstance(v, list):  # vector compare elementwise
                if len(v) != len(nv) or any(
                    abs(a - b) > 1e-6 for a, b in zip(v, nv)
                ):
                    log(f"  ABORT: field {k} differs for pk={r[pk]}")
                    return False
            elif v != nv:
                log(f"  ABORT: field {k} differs for pk={r[pk]}: "
                    f"{v!r} != {nv!r}")
                return False
    log("  gate3 random-row full-field OK (20 rows, vectors included)")

    # -- safety backup (no vectors) then swap -------------------------------
    backup = os.path.join(backup_dir, f"{name}_pre_migration_backup.jsonl")
    with open(backup, "w", encoding="utf-8") as f:
        for r in rows:
            slim = {k: v for k, v in r.items() if k != vector_field}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    log(f"  backup manifest -> {backup}")

    client.drop_collection(name)
    client.create_alias(v2, name)
    log(f"  dropped old '{name}', alias '{name}' -> '{v2}'")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", default="result")
    parser.add_argument("--force-v2", action="store_true")
    args, _ = parser.parse_known_args()

    client = MilvusClient(uri=URI)
    ok = True
    ok &= migrate(client, "memories", "id", "vector", "COSINE", args.backup_dir)
    ok &= migrate(client, "groups", "group_id", "centroid_vector", "IP",
                  args.backup_dir)

    # post-verify: schema flags + alias transparency
    for name in ("memories", "groups"):
        try:
            if not client.has_collection(name):
                continue
            desc = client.describe_collection(name)
            pkf = [f for f in desc["fields"] if f.get("is_primary")][0]
            stats = client.get_collection_stats(name)
            log(f"post-check {name}: pk={pkf['name']} "
                f"auto_id={pkf.get('auto_id')} rows~{stats.get('row_count')}")
        except Exception as e:  # noqa: BLE001 - report, don't crash the report
            log(f"post-check {name} failed: {e}")
    log("RESULT: " + ("MIGRATION OK" if ok else "MIGRATION ABORTED (nothing dropped)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
