"""Render the quantization summary as a styled PNG table for the article.

Reads data/plots/quantization_table.json (produced by plot_quantization.py)
and emits data/plots/quantization_table.png — provider-coloured row labels,
red-gradient on the binary loss column.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "data" / "plots"
TABLE_JSON = PLOTS / "quantization_table.json"

SCHEMES = ("fp32", "fp16", "int8", "int4", "binary")

PROVIDER_COLORS = {
    "openai": "#10a37f",
    "cohere": "#f06c4a",
    "google": "#4285f4",
    "qwen":   "#7b3fe4",
}

# red gradient for the loss column (lighter = small loss, darker = bigger loss)
LOSS_CMAP = LinearSegmentedColormap.from_list(
    "loss",
    ["#e8f5e9", "#fff8e1", "#ffe0b2", "#ffab91", "#ef5350"],
)


def shorten(name: str) -> str:
    """provider/base → 'provider model_short (d=DIM)'."""
    prov, base = name.split("/", 1)
    base_clean = re.sub(r"-d\d+$", "", base)
    return prov, base_clean


def loss_color(delta: float) -> str:
    """Map binary nDCG loss to a background colour.

    Δ ≤ 0.005 → green-ish; ≥ 0.060 → red-ish; linear in between.
    """
    lo, hi = 0.0, 0.07
    t = max(0.0, min(1.0, (delta - lo) / (hi - lo)))
    rgb = LOSS_CMAP(t)
    return rgb


def main() -> int:
    if not TABLE_JSON.exists():
        print(
            "[err] missing quantization_table.json — run plot_quantization.py first",
            file=sys.stderr,
        )
        return 1

    raw = json.loads(TABLE_JSON.read_text())

    # build rows sorted by fp32 nDCG (best first)
    rows = []
    for full_name, pts in raw.items():
        if "fp32" not in pts:
            continue
        prov, base = shorten(full_name)
        dim = pts["fp32"]["dim"]
        fp32_v = pts["fp32"]["ndcg@10"]
        bin_v = pts.get("binary", {}).get("ndcg@10")
        delta = (fp32_v - bin_v) if bin_v is not None else float("nan")
        rows.append({
            "provider": prov,
            "label": f"{prov} · {base} (d={dim})",
            "dim": dim,
            "scheme_vals": [pts.get(s, {}).get("ndcg@10") for s in SCHEMES],
            "delta_bin": delta,
        })
    rows.sort(key=lambda r: -r["scheme_vals"][0])  # by fp32

    headers = ["Model", "dim"] + list(SCHEMES) + ["Δ fp32→binary"]
    n_rows = len(rows)
    n_cols = len(headers)

    fig_w = 13.5
    fig_h = 1.0 + 0.55 * n_rows
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # title + subtitle above the table
    fig.text(
        0.02, 0.96,
        "Quantization sweep — nDCG@10 on SciFact (300 test queries)",
        fontsize=14, fontweight="bold", color="#222",
    )
    fig.text(
        0.02, 0.91,
        "Higher is better. Δ-column is the nDCG drop from fp32 to binary "
        "(lower is better, green = robust to binary).",
        fontsize=10, color="#555",
    )

    # build cell text + cell colours
    cell_text = []
    cell_colours = []
    for r in rows:
        provider = r["provider"]
        prov_tint = PROVIDER_COLORS.get(provider, "#888") + "26"  # 15% alpha
        row_text = [
            r["label"],
            f"{r['dim']}",
        ]
        row_colours = [prov_tint, "white"]
        for v in r["scheme_vals"]:
            row_text.append(f"{v:.4f}" if v is not None else "—")
            row_colours.append("white")
        # delta column with red gradient
        d = r["delta_bin"]
        row_text.append(f"+{d:.4f}" if d == d else "—")  # nan check
        row_colours.append(loss_color(d) if d == d else "white")
        cell_text.append(row_text)
        cell_colours.append(row_colours)

    # header row
    header_colours = ["#37474f"] * n_cols

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellColours=cell_colours,
        colColours=header_colours,
        cellLoc="center",
        loc="center",
        bbox=[0.02, 0.02, 0.96, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # styling: header text white, model column left-aligned
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cfd8dc")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(0.10)
        else:
            cell.set_height(0.075)
        if col == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.04
            cell.get_text().set_x(0.03)
        # bold the delta column
        if col == n_cols - 1 and row > 0:
            cell.get_text().set_weight("bold")

    # column widths
    col_widths = [0.36, 0.06] + [0.08] * len(SCHEMES) + [0.14]
    for col, w in enumerate(col_widths):
        for row in range(n_rows + 1):
            table[row, col].set_width(w)

    out = PLOTS / "quantization_table.png"
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())