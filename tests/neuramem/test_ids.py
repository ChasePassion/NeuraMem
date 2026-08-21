"""Snowflake id generator (core/ids.py) — the stability guarantee for
#17's upsert semantics depends on these being unique and ordered."""

from concurrent.futures import ThreadPoolExecutor

from neuramem.core.ids import new_id


def test_unique_and_monotonic_within_process():
    ids = [new_id() for _ in range(10_000)]
    assert len(set(ids)) == len(ids)
    assert all(a < b for a, b in zip(ids, ids[1:]))


def test_unique_across_threads():
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(lambda _: {new_id() for _ in range(2_000)}, range(8)))
    merged: set[int] = set()
    for batch in batches:
        merged |= batch
    assert len(merged) == 8 * 2_000


def test_int64_safe():
    assert 0 < new_id() < 2**63
