"""Analisi hard-negative sui ranking top-100 salvati da evaluate.py.

Per ogni (provider, model) calcola:
    - # hard neg medi @ top-10 e @ top-100
    - margin per query: best_pos_score - best_neg_score (top-100)
    - frazione di query con margine negativo (modello sbaglia top-1 confrontato ai positivi)

Cross-modello:
    - Jaccard dei set di hard neg @ top-10 (capire se gli errori sono condivisi)

Output:
    data/hard_neg/{provider}/{model}/mining.parquet   top-5 hard neg per query
    data/hard_neg/summary.json                        metriche aggregate e matrice Jaccard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = ROOT / "data" / "hard_neg"

K_TOP = (10, 100)
N_MINING_PER_QUERY = 5


def discover() -> list[tuple[str, str, Path]]:
    runs = []
    if not EVAL_DIR.exists():
        return runs
    for prov_dir in sorted(EVAL_DIR.iterdir()):
        if not prov_dir.is_dir():
            continue
        for mdir in sorted(prov_dir.iterdir()):
            p = mdir / "top100.parquet"
            if p.exists():
                runs.append((prov_dir.name, mdir.name, p))
    return runs


def group_by_query(path: Path) -> dict[str, list[tuple[int, str, float, bool]]]:
    cols = pq.read_table(path).to_pydict()
    per_q: dict[str, list[tuple[int, str, float, bool]]] = {}
    for i in range(len(cols["query_id"])):
        qid = cols["query_id"][i]
        per_q.setdefault(qid, []).append(
            (
                int(cols["rank"][i]),
                str(cols["doc_id"][i]),
                float(cols["score"][i]),
                bool(cols["is_relevant"][i]),
            )
        )
    for qid in per_q:
        per_q[qid].sort(key=lambda x: x[0])
    return per_q


def model_stats(per_q: dict) -> dict:
    stats: dict = {}
    for k in K_TOP:
        counts = [
            sum(1 for _, _, _, r in ranked[:k] if not r)
            for ranked in per_q.values()
        ]
        stats[f"hard_neg@{k}_mean"] = float(np.mean(counts))
        stats[f"hard_neg@{k}_p50"] = float(np.median(counts))
        stats[f"hard_neg@{k}_p95"] = float(np.percentile(counts, 95))

    margins = []
    n_no_pos = 0
    for ranked in per_q.values():
        pos = [s for _, _, s, r in ranked if r]
        neg = [s for _, _, s, r in ranked if not r]
        if not pos:
            n_no_pos += 1
            continue
        margins.append(max(pos) - (max(neg) if neg else -1.0))
    margins_arr = np.asarray(margins)
    stats["n_queries"] = len(per_q)
    stats["n_queries_no_pos_in_top100"] = n_no_pos
    stats["margin_mean"] = float(margins_arr.mean())
    stats["margin_p50"] = float(np.median(margins_arr))
    stats["margin_p05"] = float(np.percentile(margins_arr, 5))
    stats["margin_min"] = float(margins_arr.min())
    stats["pct_neg_margin"] = float((margins_arr < 0).mean())
    return stats


def hard_neg_set(per_q: dict, k: int) -> set[tuple[str, str]]:
    s = set()
    for qid, ranked in per_q.items():
        for _, did, _, r in ranked[:k]:
            if not r:
                s.add((qid, did))
    return s


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def export_mining(per_q: dict, out_path: Path, n: int) -> None:
    rows = []
    for qid, ranked in per_q.items():
        negs = [(r, d, s) for r, d, s, rel in ranked if not rel]
        for rank, did, score in negs[:n]:
            rows.append(
                {"query_id": qid, "rank": rank, "doc_id": did, "score": float(score)}
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)


def main() -> int:
    runs = discover()
    if not runs:
        print("[err] nessun ranking in data/eval/", file=sys.stderr)
        return 1

    data: dict[str, dict] = {}
    labels: dict[str, tuple[str, str]] = {}
    for prov, mdir, path in runs:
        label = f"{prov}/{mdir}"
        data[label] = group_by_query(path)
        labels[label] = (prov, mdir)

    # per-model stats + mining export
    all_stats: dict[str, dict] = {}
    for label, per_q in data.items():
        st = model_stats(per_q)
        all_stats[label] = st
        prov, mdir = labels[label]
        export_mining(per_q, OUT_DIR / prov / mdir / "mining.parquet", N_MINING_PER_QUERY)

    # print per-model
    print(f"{'='*112}")
    print("HARD NEGATIVE — per modello")
    print(f"{'='*112}")
    hdr = (
        f"  {'model':40} {'HN@10μ':>7} {'HN@100μ':>8}  "
        f"{'margin_μ':>9} {'margin_p5':>10} {'min':>7}  {'neg_margin':>10}"
    )
    print(hdr)
    print(
        f"  {'-'*40} {'-'*7} {'-'*8}  {'-'*9} {'-'*10} {'-'*7}  {'-'*10}"
    )
    for label, s in sorted(all_stats.items(), key=lambda x: -x[1]["margin_mean"]):
        print(
            f"  {label:40} "
            f"{s['hard_neg@10_mean']:>7.2f} {s['hard_neg@100_mean']:>8.2f}  "
            f"{s['margin_mean']:>9.4f} {s['margin_p05']:>10.4f} {s['margin_min']:>7.4f}  "
            f"{s['pct_neg_margin']*100:>9.1f}%"
        )

    # Jaccard matrix @ top-10
    names = sorted(data.keys())
    sets10 = {n: hard_neg_set(data[n], 10) for n in names}
    print()
    print(f"{'='*112}")
    print("JACCARD hard-neg @ top-10 (quanto modelli diversi sbagliano sulle stesse coppie q,d)")
    print(f"{'='*112}")
    short = {n: n.split("/")[0][:6] + "/" + n.split("/")[1][:10] for n in names}
    col_w = 14
    header = " " * 42 + "".join(f"{short[n]:>{col_w}}" for n in names)
    print(header)
    jaccard_mat = {}
    for a in names:
        row_vals = []
        for b in names:
            j = jaccard(sets10[a], sets10[b])
            row_vals.append(j)
            jaccard_mat.setdefault(a, {})[b] = j
        print(f"  {a:40}" + "".join(f"{v:>{col_w}.3f}" for v in row_vals))

    # save summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "per_model": all_stats,
                "jaccard_hardneg_top10": jaccard_mat,
                "k_top": list(K_TOP),
                "n_mining_per_query": N_MINING_PER_QUERY,
            },
            indent=2,
        )
    )
    print(f"\n[save] {OUT_DIR}/summary.json + mining.parquet per ciascun modello")
    return 0


if __name__ == "__main__":
    sys.exit(main())
