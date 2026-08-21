"""L1 offline consistency experiment: can cheap features reproduce the LLM's
narrative-grouping decisions?

Replays eval traces question-by-question, reconstructing the group state the
LLM saw at each decision. A pure-feature scorer (centroid cosine + session
distance + token overlap) then re-decides each event:
    merge into argmax candidate group  vs.  start a new group (cosine gate).
Agreement with the recorded LLM decision is the experiment's output.

Zero LLM calls: vectors come from the live store (memories_v2 via alias),
labels come from usage_report.assignments in the traces.

Usage:
    python scripts/l1_agreement.py --samples 0,7,8,9 \
        --traces result/locomo_full_rerun [--report result/l1]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

import dotenv

dotenv.load_dotenv(".env")

DEFAULT_TRACES = "result/locomo_full_rerun"


def load_traces(traces_dir, samples):
    """Yield per-sample ordered question records (sorted by trace timestamp)."""
    for s in samples:
        path = os.path.join(traces_dir, f"sample_{s:02d}", "eval.trace.jsonl")
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(d)
        records.sort(key=lambda d: d.get("timestamp") or "")
        yield s, records


def pull_vectors(user_ids):
    """id -> vector from the live store (alias resolves to the fixed schema)."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URI") or os.getenv("MILVUS_URL"))
    vecs = {}
    for uid in user_ids:
        rows = client.query(
            collection_name="memories",
            filter=f'user_id == "{uid}"',
            output_fields=["id", "vector"],
            limit=16384,
        )
        for r in rows:
            vecs[r["id"]] = np.asarray(r["vector"], dtype=np.float32)
    return vecs


def centroid(sum_vec, count):
    if count == 0:
        return None
    v = sum_vec / np.linalg.norm(sum_vec)
    return v


def tokens(text):
    return {w.lower().strip(".,!?;:'\"()") for w in text.split() if len(w) > 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="0,7,8,9")
    ap.add_argument("--traces", default=DEFAULT_TRACES)
    ap.add_argument("--report", default="result/l1")
    args = ap.parse_args()
    samples = [int(x) for x in args.samples.split(",")]

    vecs = pull_vectors([f"sample_{s}" for s in samples])

    events = []  # real first-time decisions
    echoes = regroups = 0
    id2meta = {}  # id -> (sample, text, session)

    for s, records in load_traces(args.traces, samples):
        mid2gid = {}  # replay state: assigned memory -> group
        groups = defaultdict(list)  # gid -> [member ids]
        for d in records:
            for m in d["retrieval"]["episodic"]:
                prov = m.get("provenance") or {}
                id2meta[m["id"]] = (
                    s, m["text"], prov.get("provenance_session") or 0
                )
            for mid, gid in ((d.get("usage_report") or {}).get("assignments") or {}).items():
                mid = int(mid)
                if mid not in vecs or mid not in id2meta:
                    continue  # semantic memory or unretrieved: outside scope
                if mid in mid2gid:
                    if mid2gid[mid] == gid:
                        echoes += 1
                    else:
                        regroups += 1
                        mid2gid[mid] = gid
                    continue
                events.append({
                    "sample": s, "mid": mid, "chosen": gid,
                    "is_new": gid not in groups,
                    "candidates": {g: list(m) for g, m in groups.items()},
                })
                mid2gid[mid] = gid
                groups[gid].append(mid)

    print(f"replay: real decisions={len(events)} echoes={echoes} regroups={regroups}")

    # -- features ------------------------------------------------------------
    def score_candidates(ev):
        s, mid = ev["sample"], ev["mid"]
        vm = vecs[mid]
        sess_m = id2meta[mid][2]
        tm = tokens(id2meta[mid][1])
        out = {}
        for g, members in ev["candidates"].items():
            vs = np.stack([vecs[x] for x in members])
            cg = centroid(vs.sum(axis=0), len(members))
            cos = float(np.dot(vm, cg)) if cg is not None else -1.0
            sess_g = [id2meta[x][2] for x in members]
            sess_gap = min(abs(sess_m - x) for x in sess_g) if sess_g else 99
            tg = set().union(*(tokens(id2meta[x][1]) for x in members))
            jac = len(tm & tg) / max(len(tm | tg), 1)
            out[g] = {"cos": cos, "sess_gap": sess_gap, "jac": jac}
        return out

    for ev in events:
        ev["feat"] = score_candidates(ev)

    # -- baselines & theta sweep ---------------------------------------------
    n_new = sum(1 for e in events if e["is_new"])
    n_merge = len(events) - n_new
    print(f"decisions: merge={n_merge} new={n_new} | always-NEW baseline={n_new/len(events):.1%}")

    # production-fidelity check: was the LLM-chosen group inside the top-8
    # cosine candidates (the store's vector pruning before the prompt)?
    topk_hits = 0
    topk_total = 0
    for ev in events:
        if ev["is_new"]:
            continue
        ranked = sorted(ev["feat"].items(), key=lambda kv: -kv[1]["cos"])[:8]
        topk_total += 1
        topk_hits += any(g == ev["chosen"] for g, _ in ranked)
    print(f"candidate retrieval: chosen group in cosine top-8: {topk_hits}/{topk_total}"
          f" ({topk_hits/max(topk_total,1):.1%})")

    # theta sweep: merge to top-cosine group iff cos >= theta else NEW
    print("\ntheta | overall | merge | new   (merge iff cos_top1 >= theta)")
    for theta in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        ok = m = mo = n = no = 0
        for ev in events:
            if not ev["feat"]:
                pred_new = True
            else:
                top = max(f["cos"] for f in ev["feat"].values())
                pred_new = top < theta
            hit = (pred_new and ev["is_new"]) or (
                not pred_new and not ev["is_new"]
                and max(ev["feat"], key=lambda g: ev["feat"][g]["cos"]) == ev["chosen"]
            )
            ok += hit
            if ev["is_new"]:
                n += 1; no += pred_new
            else:
                m += 1; mo += (not pred_new) and hit
        print(f"{theta:.2f}  | {ok/len(events):6.1%} | "
              f"{mo/max(m,1):5.1%} | {no/max(n,1):5.1%}")

    # -- cascade table: confidence = cos_top1 (coverage vs agreement) --------
    print("\nconfident band (cos_top1 >= theta, predict merge):")
    for theta in [0.70, 0.75, 0.80, 0.85, 0.90]:
        cov = corr = 0
        for ev in events:
            if not ev["feat"]:
                continue
            top_g = max(ev["feat"], key=lambda g: ev["feat"][g]["cos"])
            top = ev["feat"][top_g]["cos"]
            if top >= theta:
                cov += 1
                corr += (top_g == ev["chosen"])
        print(f"  theta={theta:.2f}: coverage={cov/len(events):5.1%} "
              f"precision={corr/max(cov,1):5.1%}")

    # -- temporal signal ------------------------------------------------------
    gaps_merge, gaps_new = [], []
    for ev in events:
        if not ev["feat"]:
            gaps_new.append(99); continue
        g = min(ev["feat"].values(), key=lambda f: f["sess_gap"])
        (gaps_merge if not ev["is_new"] else gaps_new).append(g["sess_gap"])
    def q(v, p):
        return sorted(v)[int(len(v) * p)] if v else -1
    print("\nsession gap to nearest candidate member (top-1 cosine group):")
    print(f"  merge decisions: p25={q(gaps_merge,.25)} p50={q(gaps_merge,.5)} p75={q(gaps_merge,.75)}")
    print(f"  new decisions  : p25={q(gaps_new,.25)} p50={q(gaps_new,.5)} p75={q(gaps_new,.75)}")

    os.makedirs(args.report, exist_ok=True)
    out = {
        "generated": datetime.now().isoformat(),
        "samples": samples,
        "events": len(events), "echoes": echoes, "regroups": regroups,
        "merge": n_merge, "new": n_new,
        "top8_recall": topk_hits / max(topk_total, 1),
    }
    with open(os.path.join(args.report, "l1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nreport -> {args.report}/l1_summary.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
