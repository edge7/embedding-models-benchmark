"""Tabella comparativa di costo/calls per tutti i provider su SciFact.

Usa il pricing in src/benchmark/pricing.py. Non effettua chiamate API:
stima token con char/4 e costi da listino.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmark import pricing as pricing_mod

ROOT = Path(__file__).resolve().parent.parent


def load_split_chars(split: str) -> tuple[int, int]:
    path = ROOT / "data" / "scifact" / f"{split}.jsonl"
    n = 0
    chars = 0
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            merged = (title + " " + text).strip() if title else text.strip()
            n += 1
            chars += len(merged)
    return n, chars


# batch_size noti per ciascun provider/modello
BATCH_SIZE = {
    ("openai", "text-embedding-3-large"): 2048,
    ("openai", "text-embedding-3-small"): 2048,
    ("cohere", "embed-v4.0"): 96,
    ("qwen", "text-embedding-v4"): 10,
    ("google", "gemini-embedding-001"): 1,
    ("google", "text-embedding-005"): 250,
}


def main() -> int:
    splits = {s: load_split_chars(s) for s in ("corpus", "queries")}

    print(f"{'='*110}")
    print("STIMA COSTI PER SCIFACT (BEIR)")
    print(f"{'='*110}")
    for split, (n, chars) in splits.items():
        tokens = chars // 4
        print(f"\n[{split}]  {n} items  {chars:,} char  ~{tokens:,} token (char/4)")
        print(
            f"  {'provider':10} {'model':28} {'price':>16}  "
            f"{'calls':>6}  {'cost':>10}"
        )
        print(f"  {'-'*10} {'-'*28} {'-'*16}  {'-'*6}  {'-'*10}")
        for (prov, model), pr in pricing_mod.PRICING.items():
            bs = BATCH_SIZE.get((prov, model), "?")
            calls = "?" if bs == "?" else (n + bs - 1) // bs
            cost = pr.cost(tokens, chars)
            price_str = f"${pr.price_usd}{pr.unit_label()}"
            print(
                f"  {prov:10} {model:28} {price_str:>16}  "
                f"{str(calls):>6}  ${cost:>9.4f}"
            )

    # totali corpus+queries
    tot_n = sum(n for n, _ in splits.values())
    tot_chars = sum(c for _, c in splits.values())
    tot_tokens = tot_chars // 4
    print(f"\n[TOTALE corpus+queries]  {tot_n} items  ~{tot_tokens:,} token")
    print(f"  {'provider':10} {'model':28} {'cost totale':>14}")
    print(f"  {'-'*10} {'-'*28} {'-'*14}")
    for (prov, model), pr in pricing_mod.PRICING.items():
        cost = pr.cost(tot_tokens, tot_chars)
        print(f"  {prov:10} {model:28} ${cost:>13.4f}")

    print()
    print("NOTA: prezzi indicativi da ri-verificare sulle pagine ufficiali.")
    print("      stima token = char / 4 (molto grossolana).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
