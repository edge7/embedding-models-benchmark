"""Valuta gli embedding salvati contro il test set di SciFact (BEIR).

Auto-discovery: scandisce data/embeddings/{provider}/{model_dir}/ e valuta
ogni run che ha sia corpus.parquet che queries.parquet.

Output per ogni run, in data/eval/{provider}/{model_dir}/:
    top100.parquet   righe (query_id, rank, doc_id, score, is_relevant)
    metrics.json     {n_queries, ndcg@10, recall@10, recall@100, dim}

A fine esecuzione stampa tabella comparativa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EMBED_DIR = ROOT / "data" / "embeddings"
EVAL_DIR = ROOT / "data" / "eval"
QRELS_PATH = ROOT / "data" / "scifact" / "qrels" / "test.tsv"


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        next(f)  # skip header
        for line in f:
            qid, docid, score = line.strip().split("\t")
            rel = int(score)
            if rel > 0:
                qrels.setdefault(str(qid), {})[str(docid)] = rel
    return qrels


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path)
    ids = [str(x) for x in t["id"].to_pylist()]
    embs = np.array(t["embedding"].to_pylist(), dtype=np.float32)
    return ids, embs


def l2_normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return a / norms


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked[:k]):
        r = rels.get(did, 0)
        if r > 0:
            dcg += (2.0**r - 1.0) / np.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2.0**r - 1.0) / np.log2(i + 2) for i, r in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    retrieved = set(ranked[:k])
    relevant = set(rels.keys())
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def discover_runs() -> list[tuple[str, str]]:
    found = []
    if not EMBED_DIR.exists():
        return found
    for prov_dir in sorted(EMBED_DIR.iterdir()):
        if not prov_dir.is_dir():
            continue
        for model_dir in sorted(prov_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if (
                (model_dir / "corpus.parquet").exists()
                and (model_dir / "queries.parquet").exists()
            ):
                found.append((prov_dir.name, model_dir.name))
    return found


def evaluate_one(
    provider: str, model_dir: str, qrels: dict[str, dict[str, int]]
) -> dict:
    q_path = EMBED_DIR / provider / model_dir / "queries.parquet"
    c_path = EMBED_DIR / provider / model_dir / "corpus.parquet"

    qids_all, q_emb = load_embeddings(q_path)
    cids, c_emb = load_embeddings(c_path)

    q_emb = l2_normalize(q_emb)
    c_emb = l2_normalize(c_emb)

    # restrict to queries in test qrels
    test_qids = set(qrels.keys())
    idx = [i for i, q in enumerate(qids_all) if q in test_qids]
    qids_test = [qids_all[i] for i in idx]
    q_emb_test = q_emb[idx]

    if not qids_test:
        raise RuntimeError(f"nessuna query di test trovata per {provider}/{model_dir}")

    # cosine similarity
    sim = q_emb_test @ c_emb.T  # (n_q, n_c)
    K = min(100, sim.shape[1])

    # argpartition for top-K then sort within
    top_unsorted = np.argpartition(-sim, K - 1, axis=1)[:, :K]
    rows = np.arange(top_unsorted.shape[0])[:, None]
    top_scores_unsorted = sim[rows, top_unsorted]
    order = np.argsort(-top_scores_unsorted, axis=1)
    top_idx = top_unsorted[rows, order]
    top_scores = top_scores_unsorted[rows, order]

    ndcg10, rec10, rec100 = [], [], []
    records = []
    for i, qid in enumerate(qids_test):
        ranked = [cids[j] for j in top_idx[i]]
        rels = qrels[qid]
        ndcg10.append(ndcg_at_k(ranked, rels, 10))
        rec10.append(recall_at_k(ranked, rels, 10))
        rec100.append(recall_at_k(ranked, rels, 100))
        scores_row = top_scores[i].tolist()
        for rank, (doc_id, score) in enumerate(zip(ranked, scores_row, strict=True)):
            records.append(
                {
                    "query_id": qid,
                    "rank": rank,
                    "doc_id": doc_id,
                    "score": float(score),
                    "is_relevant": doc_id in rels,
                }
            )

    eval_dir = EVAL_DIR / provider / model_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), eval_dir / "top100.parquet")

    metrics = {
        "provider": provider,
        "model_dir": model_dir,
        "n_queries": len(qids_test),
        "dim": int(q_emb.shape[1]),
        "ndcg@10": float(np.mean(ndcg10)),
        "recall@10": float(np.mean(rec10)),
        "recall@100": float(np.mean(rec100)),
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> int:
    qrels = load_qrels(QRELS_PATH)
    runs = discover_runs()
    if not runs:
        print("[err] nessun embedding trovato in data/embeddings/", file=sys.stderr)
        return 1

    print(f"[info] {len(runs)} run, {len(qrels)} test queries")
    results = []
    for prov, mdir in runs:
        print(f"  - {prov}/{mdir} ...", end=" ", flush=True)
        m = evaluate_one(prov, mdir, qrels)
        results.append(m)
        print(
            f"nDCG@10={m['ndcg@10']:.4f} "
            f"R@10={m['recall@10']:.4f} "
            f"R@100={m['recall@100']:.4f}"
        )

    print()
    print(f"{'-'*94}")
    print(
        f"  {'provider':8} {'model':36} {'dim':>5}  "
        f"{'nDCG@10':>8} {'R@10':>7} {'R@100':>7}"
    )
    print(f"  {'-'*8} {'-'*36} {'-'*5}  {'-'*8} {'-'*7} {'-'*7}")
    for m in sorted(results, key=lambda x: -x["ndcg@10"]):
        print(
            f"  {m['provider']:8} {m['model_dir']:36} {m['dim']:>5}  "
            f"{m['ndcg@10']:>8.4f} {m['recall@10']:>7.4f} {m['recall@100']:>7.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
