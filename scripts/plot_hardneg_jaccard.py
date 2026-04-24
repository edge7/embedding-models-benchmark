"""Heatmap Jaccard hard-neg@top-10 cross-model. Per articolo LinkedIn."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "data" / "hard_neg" / "summary.json"
OUT = ROOT / "data" / "plots" / "hardneg_jaccard.png"

SHORT = {
    "cohere/embed-v4.0": "Cohere v4\n(d=1536)",
    "google/gemini-embedding-001-d3072": "Gemini\n(d=3072)",
    "google/gemini-embedding-001-d768": "Gemini\n(d=768)",
    "google/text-embedding-005-d768": "Google\ntext-005\n(d=768)",
    "openai/text-embedding-3-large-d3072": "OpenAI\n3-large\n(d=3072)",
    "qwen/text-embedding-v4": "Qwen v4\n(d=1024)",
    "qwen/text-embedding-v4-d2048": "Qwen v4\n(d=2048)",
}
# order: put providers contiguous for readability
ORDER = [
    "google/gemini-embedding-001-d768",
    "google/gemini-embedding-001-d3072",
    "openai/text-embedding-3-large-d3072",
    "cohere/embed-v4.0",
    "google/text-embedding-005-d768",
    "qwen/text-embedding-v4-d2048",
    "qwen/text-embedding-v4",
]


def main() -> int:
    s = json.loads(SUMMARY.read_text())
    mat = s["jaccard_hardneg_top10"]
    labels = [k for k in ORDER if k in mat]
    n = len(labels)
    M = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            M[i, j] = mat[a][b]

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    fig.subplots_adjust(top=0.88, left=0.18, right=0.95, bottom=0.14)
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1.0, aspect="equal")

    ticks = [SHORT.get(lbl, lbl) for lbl in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ticks, fontsize=9, rotation=0, ha="center")
    ax.set_yticklabels(ticks, fontsize=9)

    # cell annotations
    for i in range(n):
        for j in range(n):
            val = M[i, j]
            color = "white" if val > 0.5 else "#1a1a1a"
            weight = "bold" if i == j else "normal"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color=color, fontsize=10.5, fontweight=weight,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.042, pad=0.04)
    cbar.set_label("Jaccard overlap of hard-neg sets @ top-10", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        "How similar are the errors across models?",
        fontsize=14, fontweight="bold", x=0.04, y=0.975, ha="left",
    )
    fig.text(
        0.04, 0.935,
        "Jaccard of (query, doc) hard-negative pairs in top-10 — SciFact / BEIR",
        fontsize=9.5, color="#555", ha="left",
    )

    ax.tick_params(which="both", length=0)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[save] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
