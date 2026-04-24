"""Bar chart neg_margin % per modello.

neg_margin% = frazione di query in cui il miglior falso-positivo
ha score superiore al MIGLIORE vero-positivo nei top-100. Lower is better.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "data" / "hard_neg" / "summary.json"
OUT = ROOT / "data" / "plots" / "neg_margin.png"

PROVIDER_COLORS = {
    "openai": "#10a37f",
    "cohere": "#f06c4a",
    "google": "#4285f4",
    "qwen": "#7b3fe4",
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


def main() -> int:
    s = json.loads(SUMMARY.read_text())
    per_model = s["per_model"]
    rows = [
        {
            "label": SHORT.get(k, k),
            "provider": k.split("/")[0],
            "pct": stats["pct_neg_margin"] * 100,
        }
        for k, stats in per_model.items()
    ]
    rows.sort(key=lambda r: r["pct"])  # best first

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(top=0.84, left=0.30, right=0.94, bottom=0.13)

    labels = [r["label"] for r in rows]
    vals = [r["pct"] for r in rows]
    colors = [PROVIDER_COLORS.get(r["provider"], "#888") for r in rows]

    ax.barh(
        range(len(rows)), vals,
        color=colors, alpha=0.9, edgecolor="white", linewidth=1.5,
    )
    for i, val in enumerate(vals):
        ax.text(
            val + 0.6, i, f"{val:.1f}%",
            va="center", fontsize=10.5, color="#222", fontweight="medium",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Queries with a catastrophic miss (%)", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(True, alpha=0.25, axis="x", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    fig.suptitle(
        "When a model is wrong, how catastrophic is it?",
        fontsize=14, fontweight="bold", x=0.04, y=0.975, ha="left",
    )
    fig.text(
        0.04, 0.93,
        "% of queries where a false positive outranks every true positive in top-100 — lower is better",
        fontsize=9.5, color="#555", ha="left",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[save] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
