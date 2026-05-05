"""Quantize already-computed embeddings (fp16, int8, int4, binary).

For each existing run under data/embeddings/{provider}/{model_dir}/, produces
a sibling run under data/embeddings/{provider}/{model_dir}-{scheme}/ with the
embeddings dequantized back to float (so scripts/evaluate.py can consume it).

The "actual" compression isn't on disk (we still store float for evaluation
convenience) — the *information loss* matches a real on-disk quantized vector.
For each scheme we record analytical bytes/vector in manifest.json.

Quantization schemes (all symmetric per-dimension, scale fitted on the corpus):

    fp16     half-precision; effectively lossless. dim * 2 bytes/vec.
    int8     [-127, 127] per dim. dim * 1 bytes/vec.
    int4     [-7, 7]    per dim. dim * 0.5 bytes/vec.
    binary   sign(x).            dim / 8 bytes/vec.

The same scale per dim, fitted on the corpus, is used for queries — this is how
production systems work (the index is built once, queries are quantized at
search time using the index's scale).

Usage:
    uv run python scripts/quantize.py            # process all runs
    uv run python scripts/quantize.py --schemes fp16,binary
    uv run python scripts/quantize.py --provider cohere
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
EMBED_DIR = ROOT / "data" / "embeddings"

SCHEMES = ("fp16", "int8", "int4", "binary")


def bytes_per_vec(scheme: str, dim: int) -> float:
    if scheme == "fp32":
        return dim * 4.0
    if scheme == "fp16":
        return dim * 2.0
    if scheme == "int8":
        return dim * 1.0
    if scheme == "int4":
        return dim * 0.5
    if scheme == "binary":
        return dim / 8.0
    raise ValueError(scheme)


def fit_scale_symmetric(corpus: np.ndarray, n_levels: int) -> np.ndarray:
    """Per-dimension symmetric scale: |max| / level_max."""
    abs_max = np.maximum(np.abs(corpus.max(axis=0)), np.abs(corpus.min(axis=0)))
    abs_max = np.where(abs_max == 0.0, 1.0, abs_max)
    return abs_max / n_levels  # shape (dim,)


def quantize_int(x: np.ndarray, scale: np.ndarray, n_levels: int) -> np.ndarray:
    """Symmetric scalar quantization → dequantize back to float32.

    Information loss is what evaluate.py will measure.
    """
    q = np.round(x / scale).clip(-n_levels, n_levels)
    return (q * scale).astype(np.float32)


def quantize_fp16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16).astype(np.float32)


def quantize_binary(x: np.ndarray) -> np.ndarray:
    """Sign quantization: each dim becomes ±1.

    Cosine similarity on {-1,+1} vectors equals (D - 2·Hamming(x,y)) / D, so
    ranking with cosine is identical to ranking with Hamming distance — what
    a real binary index would compute.
    """
    s = np.sign(x).astype(np.float32)
    # zeros (rare) → +1 to avoid degenerate vectors
    s[s == 0.0] = 1.0
    return s


def discover_runs() -> list[tuple[str, str]]:
    """Return (provider, model_dir) for every fp32 run on disk.

    Skips runs whose model_dir already carries a quantization suffix (so we
    don't quantize-the-quantized).
    """
    found = []
    if not EMBED_DIR.exists():
        return found
    suffixes = tuple(f"-{s}" for s in SCHEMES)
    for prov_dir in sorted(EMBED_DIR.iterdir()):
        if not prov_dir.is_dir():
            continue
        # skip fusion outputs (handled separately if ever needed)
        if prov_dir.name.startswith("fusion-"):
            continue
        for mdir in sorted(prov_dir.iterdir()):
            if not mdir.is_dir():
                continue
            if mdir.name.endswith(suffixes):
                continue
            if (mdir / "corpus.parquet").exists() and (mdir / "queries.parquet").exists():
                found.append((prov_dir.name, mdir.name))
    return found


def load_parquet(path: Path) -> tuple[list[str], np.ndarray]:
    t = pq.read_table(path)
    ids = [str(x) for x in t["id"].to_pylist()]
    embs = np.array(t["embedding"].to_pylist(), dtype=np.float32)
    return ids, embs


def save_parquet(path: Path, ids: list[str], embs: np.ndarray) -> None:
    rows = [
        {"id": i, "embedding": v.tolist()}
        for i, v in zip(ids, embs, strict=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def process_run(provider: str, model_dir: str, schemes: list[str]) -> dict:
    src = EMBED_DIR / provider / model_dir
    cids, corpus = load_parquet(src / "corpus.parquet")
    qids, queries = load_parquet(src / "queries.parquet")
    dim = corpus.shape[1]
    print(
        f"  {provider}/{model_dir}  dim={dim}  "
        f"corpus={corpus.shape[0]}  queries={queries.shape[0]}"
    )

    summary = {
        "provider": provider,
        "src_model_dir": model_dir,
        "dim": int(dim),
        "schemes": {},
    }

    # fp32 reference (just compute its byte size for the manifest)
    summary["schemes"]["fp32"] = {
        "bytes_per_vec": bytes_per_vec("fp32", dim),
        "compression_vs_fp32": 1.0,
    }

    for scheme in schemes:
        out_model_dir = f"{model_dir}-{scheme}"
        out = EMBED_DIR / provider / out_model_dir

        if scheme == "fp16":
            c_q = quantize_fp16(corpus)
            q_q = quantize_fp16(queries)
        elif scheme == "binary":
            c_q = quantize_binary(corpus)
            q_q = quantize_binary(queries)
        elif scheme in ("int8", "int4"):
            n_levels = 127 if scheme == "int8" else 7
            scale = fit_scale_symmetric(corpus, n_levels)
            c_q = quantize_int(corpus, scale, n_levels)
            q_q = quantize_int(queries, scale, n_levels)
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        save_parquet(out / "corpus.parquet", cids, c_q)
        save_parquet(out / "queries.parquet", qids, q_q)

        bpv = bytes_per_vec(scheme, dim)
        meta = {
            "scheme": scheme,
            "dim": int(dim),
            "bytes_per_vec": bpv,
            "compression_vs_fp32": (dim * 4.0) / bpv,
            "src": f"{provider}/{model_dir}",
        }
        (out / "quant_manifest.json").write_text(json.dumps(meta, indent=2))
        summary["schemes"][scheme] = {
            "bytes_per_vec": bpv,
            "compression_vs_fp32": meta["compression_vs_fp32"],
        }

        print(
            f"    [{scheme:>6}] {bpv:>8.1f} B/vec  "
            f"({meta['compression_vs_fp32']:>5.1f}× compression)  → {out.name}"
        )

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--schemes", default=",".join(SCHEMES),
        help=f"csv among {SCHEMES}",
    )
    ap.add_argument("--provider", help="filter by provider name")
    args = ap.parse_args()

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    bad = [s for s in schemes if s not in SCHEMES]
    if bad:
        print(f"[err] unknown schemes: {bad}", file=sys.stderr)
        return 1

    runs = discover_runs()
    if args.provider:
        runs = [r for r in runs if r[0] == args.provider]
    if not runs:
        print("[err] no runs found", file=sys.stderr)
        return 1

    print(f"[info] {len(runs)} runs × {len(schemes)} schemes")
    summaries = []
    for prov, mdir in runs:
        summaries.append(process_run(prov, mdir, schemes))

    manifest = ROOT / "data" / "embeddings" / "quant_summary.json"
    manifest.write_text(json.dumps(summaries, indent=2))
    print(f"\n[save] {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())