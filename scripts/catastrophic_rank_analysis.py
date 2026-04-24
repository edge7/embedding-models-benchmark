"""Rank del miglior doc relevant sulle catastrophic queries.

Per ogni modello, considera solo le query con margin<0 (catastrofiche: un falso
positivo batte ogni vero positivo nei top-100). Per quelle query, estrae il
rank 1-indexed del miglior vero positivo presente nei top-100.

Output:
    results/catastrophic_rank_stats.json   stats per modello
    results/catastrophic_rank_bar.png      bar chart median rank
    results/catastrophic_rank_box.png      box plot distribuzione rank
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
OUT = ROOT / "results"

PROVIDER_COLORS = {
    "openai": "#10a37f",
    "cohere": "#f06c4a",
    "google": "#4285f4",
    "qwen":   "#7b3fe4",
}

SHORT = {
    "cohere/embed-v4.0":                    "Cohere embed-v4.0 (d=1536)",
    "google/gemini-embedding-001-d3072":    "Gemini-embedding-001 (d=3072)",
    "google/gemini-embedding-001-d768":     "Gemini-embedding-001 (d=768)",
    "google/text-embedding-005-d768":       "Google text-embedding-005 (d=768)",
    "openai/text-embedding-3-large-d3072":  "OpenAI text-embedding-3-large (d=3072)",
    "qwen/text-embedding-v4":               "Qwen text-embedding-v4 (d=1024)",
    "qwen/text-embedding-v4-d2048":         "Qwen text-embedding-v4 (d=2048)",
}


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


def catastrophic_ranks(path: Path) -> tuple[list[int], int]:
    """Per la run a `path`, ritorna:
        ranks_1idx   lista di rank (1-indexed) del miglior vero-positivo
                     per ogni query catastrofica
        total_qwp    numero di query con almeno un positivo in top-100
    """
    cols = pq.read_table(path).to_pydict()

    per_q: dict[str, list[tuple[int, float, bool]]] = {}
    for i in range(len(cols["query_id"])):
        qid = cols["query_id"][i]
        per_q.setdefault(qid, []).append(
            (
                int(cols["rank"][i]),
                float(cols["score"][i]),
                bool(cols["is_relevant"][i]),
            )
        )
    for qid in per_q:
        per_q[qid].sort(key=lambda r: r[0])

    ranks_1idx: list[int] = []
    total_qwp = 0
    for ranked in per_q.values():
        pos = [(r, s) for r, s, rel in ranked if rel]
        if not pos:
            continue
        total_qwp += 1
        neg_scores = [s for _, s, rel in ranked if not rel]
        best_pos_score = max(s for _, s in pos)
        best_neg_score = max(neg_scores) if neg_scores else -float("inf")
        if best_pos_score < best_neg_score:
            # catastrophic — record rank of top-ranked positive (lowest 0-idx rank)
            best_pos_rank_0 = min(r for r, _ in pos)
            ranks_1idx.append(best_pos_rank_0 + 1)
    return ranks_1idx, total_qwp


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.25, axis="x", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", length=0)


def plot_bar(stats: dict, out_path: Path) -> None:
    rows = [
        (k, v["median_rank_of_best_relevant"], v["n_catastrophic"], v["provider"])
        for k, v in stats.items()
        if v["median_rank_of_best_relevant"] is not None
    ]
    rows.sort(key=lambda r: r[1])  # best first

    labels = [SHORT.get(r[0], r[0]) for r in rows]
    vals   = [r[1] for r in rows]
    counts = [r[2] for r in rows]
    colors = [PROVIDER_COLORS.get(r[3], "#888") for r in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(top=0.83, left=0.32, right=0.94, bottom=0.13)

    ax.barh(
        range(len(rows)), vals,
        color=colors, alpha=0.9, edgecolor="white", linewidth=1.5,
    )
    for i, (val, cnt) in enumerate(zip(vals, counts)):
        ax.text(
            val + max(vals) * 0.015, i,
            f"rank {val:.1f}  (n={cnt})",
            va="center", fontsize=10.5, color="#222", fontweight="medium",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Median rank of best relevant document", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.30)
    ax.invert_yaxis()
    _style_axes(ax)

    fig.suptitle(
        "When the retriever fails: how deep is the right document buried?",
        fontsize=13, fontweight="bold", x=0.04, y=0.975, ha="left",
    )
    fig.text(
        0.04, 0.925,
        "Median rank of best relevant document on catastrophic queries — lower is better",
        fontsize=9.5, color="#555", ha="left",
    )

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_box(stats: dict, per_model_ranks: dict, out_path: Path) -> None:
    models = [
        k for k, ranks in per_model_ranks.items()
        if ranks and stats[k]["median_rank_of_best_relevant"] is not None
    ]
    models.sort(key=lambda k: stats[k]["median_rank_of_best_relevant"])
    labels = [SHORT.get(k, k) for k in models]
    data   = [per_model_ranks[k] for k in models]
    colors = [PROVIDER_COLORS.get(k.split("/")[0], "#888") for k in models]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(top=0.83, left=0.32, right=0.94, bottom=0.13)

    bp = ax.boxplot(
        data,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(
            marker="o", markersize=3.5,
            markerfacecolor="#888", markeredgecolor="none", alpha=0.55,
        ),
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(color="#666", linewidth=1),
        capprops=dict(color="#666", linewidth=1),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor("white")
        patch.set_linewidth(1.5)

    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Rank of best relevant document (catastrophic queries)", fontsize=11)
    ax.invert_yaxis()
    _style_axes(ax)

    fig.suptitle(
        "When the retriever fails: how deep is the right document buried?",
        fontsize=13, fontweight="bold", x=0.04, y=0.975, ha="left",
    )
    fig.text(
        0.04, 0.925,
        "Distribution of best-relevant rank on catastrophic queries — lower and tighter is better",
        fontsize=9.5, color="#555", ha="left",
    )

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    runs = discover()
    if not runs:
        print("[err] no eval data in data/eval/", file=sys.stderr)
        return 1

    stats: dict[str, dict] = {}
    per_model_ranks: dict[str, list[int]] = {}

    for prov, mdir, path in runs:
        key = f"{prov}/{mdir}"
        ranks, total_qwp = catastrophic_ranks(path)
        per_model_ranks[key] = ranks
        arr = np.asarray(ranks, dtype=float)
        st: dict = {
            "provider": prov,
            "model_dir": mdir,
            "total_queries_with_positives_in_top100": total_qwp,
            "n_catastrophic": len(ranks),
            "pct_catastrophic": float(len(ranks) / total_qwp) if total_qwp else 0.0,
        }
        if len(ranks) > 0:
            st.update({
                "median_rank_of_best_relevant": float(np.median(arr)),
                "mean_rank_of_best_relevant":   float(arr.mean()),
                "p90_rank_of_best_relevant":    float(np.percentile(arr, 90)),
                "min_rank_of_best_relevant":    int(arr.min()),
                "max_rank_of_best_relevant":    int(arr.max()),
            })
        else:
            st.update({
                "median_rank_of_best_relevant": None,
                "mean_rank_of_best_relevant":   None,
                "p90_rank_of_best_relevant":    None,
                "min_rank_of_best_relevant":    None,
                "max_rank_of_best_relevant":    None,
            })
        stats[key] = st

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "catastrophic_rank_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[save] {OUT / 'catastrophic_rank_stats.json'}")

    plot_bar(stats, OUT / "catastrophic_rank_bar.png")
    print(f"[save] {OUT / 'catastrophic_rank_bar.png'}")

    plot_box(stats, per_model_ranks, OUT / "catastrophic_rank_box.png")
    print(f"[save] {OUT / 'catastrophic_rank_box.png'}")

    # summary
    print()
    print(f"  {'model':40} {'n_cat':>5}  {'median':>6}  {'mean':>6}  {'p90':>6}")
    print(f"  {'-'*40} {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}")
    ordered = sorted(
        stats.items(),
        key=lambda x: x[1]["median_rank_of_best_relevant"] or float("inf"),
    )
    for k, st in ordered:
        if st["median_rank_of_best_relevant"] is None:
            continue
        print(
            f"  {k:40} {st['n_catastrophic']:>5}  "
            f"{st['median_rank_of_best_relevant']:>6.1f}  "
            f"{st['mean_rank_of_best_relevant']:>6.1f}  "
            f"{st['p90_rank_of_best_relevant']:>6.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
