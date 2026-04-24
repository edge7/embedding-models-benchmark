"""Scatter cost-vs-quality per ogni (provider, model). Output PNG per LinkedIn.

Costi REALI calcolati dai _runs.jsonl (token/char fatturati dall'API), moltiplicati
per il prezzo in src/benchmark/pricing.py. Se un modello ha più sessioni (resume),
sommiamo i batch di tutte le sessioni.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmark import pricing as pricing_mod

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
EMBED_DIR = ROOT / "data" / "embeddings"
OUT = ROOT / "data" / "plots" / "cost_vs_quality.png"

PROVIDER_COLORS = {
    "openai": "#10a37f",
    "cohere": "#f06c4a",
    "google": "#4285f4",
    "qwen": "#7b3fe4",
}


def strip_dim_suffix(name: str) -> str:
    return re.sub(r"-d\d+$", "", name)


def sum_from_runs(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    tot_tok, tot_chr = 0, 0
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            tot_tok += r.get("input_tokens") or 0
            tot_chr += r.get("input_chars") or 0
    return tot_tok, tot_chr


def compute_cost(provider: str, model: str, model_dir: str) -> float | None:
    pr = pricing_mod.get(provider, model)
    if pr is None:
        return None
    tok_tot, chr_tot = 0, 0
    for split in ("corpus", "queries"):
        t, c = sum_from_runs(EMBED_DIR / provider / model_dir / f"{split}_runs.jsonl")
        tok_tot += t
        chr_tot += c
    return pr.cost(tok_tot, chr_tot)


def collect_points() -> list[dict]:
    points = []
    for prov_dir in sorted(EVAL_DIR.iterdir()):
        if not prov_dir.is_dir():
            continue
        for mdir in sorted(prov_dir.iterdir()):
            metrics = mdir / "metrics.json"
            if not metrics.exists():
                continue
            m = json.loads(metrics.read_text())
            base = strip_dim_suffix(mdir.name)
            cost = compute_cost(prov_dir.name, base, mdir.name)
            if cost is None:
                continue
            points.append(
                {
                    "provider": prov_dir.name,
                    "model_base": base,
                    "model_dir": mdir.name,
                    "dim": m["dim"],
                    "ndcg": m["ndcg@10"],
                    "cost": cost,
                }
            )
    return points


def pareto_front(points: list[dict]) -> list[dict]:
    """Cost asc, ndcg desc as tie-breaker → a parità di costo vince il top quality."""
    sorted_pts = sorted(points, key=lambda p: (p["cost"], -p["ndcg"]))
    pareto: list[dict] = []
    best = -1.0
    for p in sorted_pts:
        if p["ndcg"] > best:
            pareto.append(p)
            best = p["ndcg"]
    return pareto


def annotation_xy(p: dict) -> tuple[float, float]:
    """Manual offsets per modello. Direzioni opposte per le coppie same-cost."""
    offsets = {
        # qwen pair @ same cost — una a sinistra, una a destra
        "text-embedding-v4":             (-125,  30),
        "text-embedding-v4-d2048":       (  35, -30),
        # google pair @ same cost — una a sinistra, una a destra
        "gemini-embedding-001-d3072":    (-180,  22),
        "gemini-embedding-001-d768":     (  22, -18),
        # isolati
        "text-embedding-005-d768":       (-150,  -5),
        "embed-v4.0":                    (  22, -18),
        "text-embedding-3-large-d3072":  (  22,  14),
    }
    return offsets.get(p["model_dir"], (10, 6))


def main() -> int:
    points = collect_points()
    if not points:
        print("[err] no points found", file=sys.stderr)
        return 1

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    # Pareto frontier line (behind points)
    pareto = pareto_front(points)
    ax.plot(
        [p["cost"] for p in pareto],
        [p["ndcg"] for p in pareto],
        linestyle="--", color="#888", linewidth=1.5, alpha=0.55, zorder=1,
        label="Pareto frontier",
    )

    # identify dominated points (not in Pareto)
    pareto_dirs = {p["model_dir"] for p in pareto}

    # scatter points
    for p in points:
        c = PROVIDER_COLORS.get(p["provider"], "#444")
        is_pareto = p["model_dir"] in pareto_dirs
        size = 80 + p["dim"] / 8
        if is_pareto:
            ax.scatter(
                p["cost"], p["ndcg"],
                s=size, c=c, edgecolors="white", linewidth=2.2,
                alpha=0.92, zorder=3,
            )
        else:
            # dominated: hollow marker
            ax.scatter(
                p["cost"], p["ndcg"],
                s=size * 0.6, facecolors="none",
                edgecolors=c, linewidth=2.0,
                alpha=0.75, zorder=3,
            )

        dx, dy = annotation_xy(p)
        label = f"{p['model_base']}\n(d={p['dim']})"
        # arrow from label to point when label is far away
        use_arrow = abs(dx) > 40 or abs(dy) > 25
        arrowprops = (
            dict(arrowstyle="-", color="#aaa", lw=0.8,
                 connectionstyle="arc3,rad=0.1")
            if use_arrow else None
        )
        ax.annotate(
            label, xy=(p["cost"], p["ndcg"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=9, color="#222",
            ha="left" if dx >= 0 else "right",
            va="center",
            arrowprops=arrowprops,
        )

    ax.set_xlabel("Cost to embed corpus + queries  (USD, actual billed)", fontsize=11)
    ax.set_ylabel("nDCG@10 on SciFact test set", fontsize=11)
    ax.set_title(
        "Dense retrieval APIs: cost vs retrieval quality",
        fontsize=14, fontweight="bold", pad=14, loc="left",
    )
    ax.text(
        0.0, 1.02,
        "SciFact / BEIR — 5,183 docs, 300 test queries — April 2026",
        transform=ax.transAxes, fontsize=9.5, color="#555",
    )

    ax.grid(True, alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # extend axes for label room (extra left/right for offset labels)
    xs = [p["cost"] for p in points]
    ys = [p["ndcg"] for p in points]
    xrange = max(xs) - min(xs)
    ax.set_xlim(min(xs) - xrange * 0.22, max(xs) + xrange * 0.18)
    ax.set_ylim(min(ys) - 0.04, max(ys) + 0.04)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=name)
        for name, c in PROVIDER_COLORS.items()
    ]
    legend_elements.append(
        Line2D([0], [0], linestyle="--", color="#888", linewidth=1.5, label="Pareto frontier")
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#666', markeredgewidth=1.6, markersize=9,
               label="dominated (same cost,\nworse quality)")
    )
    ax.legend(
        handles=legend_elements, loc="lower right",
        title="Provider", frameon=True, framealpha=0.95, fontsize=9,
    )

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[save] {OUT}")

    # also print summary table
    print()
    print(f"  {'provider':8} {'model':36} {'dim':>5}  {'cost':>7}  {'nDCG@10':>8}  {'pareto':>7}")
    print(f"  {'-'*8} {'-'*36} {'-'*5}  {'-'*7}  {'-'*8}  {'-'*7}")
    pareto_dirs = {p["model_dir"] for p in pareto}
    for p in sorted(points, key=lambda x: x["cost"]):
        marker = " YES" if p["model_dir"] in pareto_dirs else ""
        print(
            f"  {p['provider']:8} {p['model_dir']:36} {p['dim']:>5}  "
            f"${p['cost']:>6.4f}  {p['ndcg']:>8.4f}  {marker:>7}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
