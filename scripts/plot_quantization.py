"""Plot storage (bytes/vector) vs retrieval quality (nDCG@10).

For each (provider, model) connect a polyline through its 5 quantization
schemes: fp32 → fp16 → int8 → int4 → binary. Colour by provider, marker
shape by scheme. Output:

    data/plots/quantization_pareto.png
    data/plots/quantization_table.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"
EMBED_DIR = ROOT / "data" / "embeddings"
PLOTS = ROOT / "data" / "plots"

SCHEMES = ("fp32", "fp16", "int8", "int4", "binary")
SCHEME_MARKER = {
    "fp32":   "o",
    "fp16":   "s",
    "int8":   "D",
    "int4":   "v",
    "binary": "X",
}

PROVIDER_COLORS = {
    "openai": "#10a37f",
    "cohere": "#f06c4a",
    "google": "#4285f4",
    "qwen":   "#7b3fe4",
}

DIM_FROM_NAME = re.compile(r"-d(\d+)")


def base_and_scheme(model_dir: str) -> tuple[str, str]:
    """Split 'embed-v4.0-int8' → ('embed-v4.0', 'int8'). Default scheme = fp32."""
    for s in ("fp16", "int8", "int4", "binary"):
        suf = f"-{s}"
        if model_dir.endswith(suf):
            return model_dir[: -len(suf)], s
    return model_dir, "fp32"


def model_dim(model_dir_base: str, fallback_dim: int) -> int:
    m = DIM_FROM_NAME.search(model_dir_base)
    return int(m.group(1)) if m else fallback_dim


def bytes_per_vec(scheme: str, dim: int) -> float:
    return {
        "fp32":   dim * 4.0,
        "fp16":   dim * 2.0,
        "int8":   dim * 1.0,
        "int4":   dim * 0.5,
        "binary": dim / 8.0,
    }[scheme]


def short_label(provider: str, base: str, dim: int | None = None) -> str:
    base_clean = re.sub(r"-d\d+$", "", base)
    if dim is None:
        return f"{provider} {base_clean}"
    return f"{provider} {base_clean} (d={dim})"


def collect() -> dict:
    """Return {(provider, base): {scheme: {ndcg, dim, bytes}}}."""
    data: dict = {}
    if not EVAL_DIR.exists():
        return data
    for prov_dir in sorted(EVAL_DIR.iterdir()):
        if not prov_dir.is_dir() or prov_dir.name.startswith("fusion"):
            continue
        for mdir in sorted(prov_dir.iterdir()):
            mp = mdir / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text())
            base, scheme = base_and_scheme(mdir.name)
            dim = int(m["dim"])
            key = (prov_dir.name, base)
            data.setdefault(key, {})[scheme] = {
                "ndcg@10": float(m["ndcg@10"]),
                "recall@10": float(m["recall@10"]),
                "recall@100": float(m["recall@100"]),
                "dim": dim,
                "bytes_per_vec": bytes_per_vec(scheme, dim),
            }
    return data


def plot(data: dict, out_path: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # sort models by fp32 nDCG so legend is stable / strongest first
    keys = sorted(
        data.keys(),
        key=lambda k: -data[k].get("fp32", {}).get("ndcg@10", 0.0),
    )

    for prov, base in keys:
        pts = data[(prov, base)]
        xs, ys, markers = [], [], []
        for s in SCHEMES:
            if s in pts:
                xs.append(pts[s]["bytes_per_vec"])
                ys.append(pts[s]["ndcg@10"])
                markers.append(s)
        if len(xs) < 2:
            continue
        c = PROVIDER_COLORS.get(prov, "#444")
        ax.plot(xs, ys, color=c, alpha=0.55, linewidth=1.4, zorder=2)
        for x, y, s in zip(xs, ys, markers, strict=True):
            ax.scatter(
                x, y, s=80, marker=SCHEME_MARKER[s],
                color=c, edgecolors="white", linewidth=1.4, zorder=3,
            )
        # label at fp32 (rightmost) end, with manual offsets for known clashes
        if "fp32" in pts:
            # default: small horizontal offset, vertically centered
            dx, dy = 8, 0
            # Gemini pair sits ~2pt apart on y → split vertically
            if base == "gemini-embedding-001-d768":
                dx, dy = 8, 8
            elif base == "gemini-embedding-001-d3072":
                dx, dy = 8, -8
            # Qwen pair: minor split for safety
            elif base == "text-embedding-v4-d2048":
                dx, dy = 8, 6
            elif base == "text-embedding-v4":
                dx, dy = 8, -6
            ax.annotate(
                short_label(prov, base, pts["fp32"]["dim"]),
                xy=(pts["fp32"]["bytes_per_vec"], pts["fp32"]["ndcg@10"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color="#222", va="center",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Bytes per vector (log scale)", fontsize=11)
    ax.set_ylabel("nDCG@10 on SciFact test set", fontsize=11)
    ax.set_title(
        "Quantization vs retrieval quality — how small can your embeddings get?",
        fontsize=14, fontweight="bold", pad=14, loc="left",
    )
    ax.text(
        0.0, 1.02,
        "SciFact / BEIR — 5,183 docs, 300 test queries — fp32 → fp16 → int8 → int4 → binary",
        transform=ax.transAxes, fontsize=9.5, color="#555",
    )

    ax.grid(True, which="both", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legends: scheme markers + provider colors
    scheme_handles = [
        Line2D([0], [0], marker=SCHEME_MARKER[s], color="#666",
               markerfacecolor="#bbb", markeredgecolor="white", markersize=9,
               linestyle="", label=s)
        for s in SCHEMES
    ]
    provider_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markersize=10, label=name)
        for name, c in PROVIDER_COLORS.items()
    ]
    # Make room above the highest data point and put both legends there.
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.18)

    leg1 = ax.legend(
        handles=scheme_handles, loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        title="Scheme", frameon=True, framealpha=0.95, fontsize=9, ncol=5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=provider_handles, loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        title="Provider", frameon=True, framealpha=0.95, fontsize=9, ncol=4,
    )

    # widen x-axis on the right so model labels at fp32 don't get clipped
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 3.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[save] {out_path}")


def print_table(data: dict) -> None:
    keys = sorted(
        data.keys(),
        key=lambda k: -data[k].get("fp32", {}).get("ndcg@10", 0.0),
    )
    print()
    print(
        f"  {'model':44} {'dim':>5}  "
        + "  ".join(f"{s:>10}" for s in SCHEMES)
        + "    Δ(fp→bin)"
    )
    print(
        f"  {'-'*44} {'-'*5}  "
        + "  ".join("-" * 10 for _ in SCHEMES)
        + "  " + "-" * 10
    )
    for prov, base in keys:
        pts = data[(prov, base)]
        if "fp32" not in pts:
            continue
        dim = pts["fp32"]["dim"]
        cells = []
        for s in SCHEMES:
            v = pts.get(s, {}).get("ndcg@10")
            cells.append(f"{v:>10.4f}" if v is not None else f"{'-':>10}")
        delta = pts["fp32"]["ndcg@10"] - pts.get("binary", {}).get(
            "ndcg@10", float("nan")
        )
        print(
            f"  {short_label(prov, base, dim):44} {dim:>5}  "
            + "  ".join(cells)
            + f"  {delta:>+10.4f}"
        )


def main() -> int:
    data = collect()
    if not data:
        print("[err] no eval data", file=sys.stderr)
        return 1

    PLOTS.mkdir(parents=True, exist_ok=True)
    table = {
        f"{prov}/{base}": pts for (prov, base), pts in data.items()
    }
    (PLOTS / "quantization_table.json").write_text(json.dumps(table, indent=2))
    print(f"[save] {PLOTS / 'quantization_table.json'}")

    plot(data, PLOTS / "quantization_pareto.png")
    print_table(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())