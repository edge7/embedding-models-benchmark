"""Embedding di SciFact → parquet su disco, con resume e save per-batch.

Esempi:
    uv run python scripts/embed.py --provider cohere --split queries --dry-run
    uv run python scripts/embed.py --provider qwen   --split corpus  --limit 50
    uv run python scripts/embed.py --provider cohere --split corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm

from benchmark import pricing as pricing_mod

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def load_split(split: str) -> list[dict]:
    path = ROOT / "data" / "scifact" / f"{split}.jsonl"
    items: list[dict] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            merged = (title + " " + text).strip() if title else text.strip()
            items.append({"id": str(obj["_id"]), "text": merged})
    return items


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CohereProvider:
    name = "cohere"
    model = "embed-v4.0"
    batch_size = 96
    # trial key: 100k tokens/min osservato dal 429 error (aprile 2026)
    tpm_limit = 100_000

    def __init__(self) -> None:
        import cohere

        self.client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

    def embed(
        self, texts: list[str], role: str
    ) -> tuple[list[list[float]], dict]:
        input_type = "search_query" if role == "queries" else "search_document"
        t0 = time.perf_counter()
        resp = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            embedding_types=["float"],
        )
        latency = time.perf_counter() - t0
        toks = None
        if resp.meta and resp.meta.billed_units:
            toks = resp.meta.billed_units.input_tokens
        usage = {"input_tokens": toks, "latency_s": latency}
        return resp.embeddings.float_, usage


class GoogleProvider:
    name = "google"
    model = "gemini-embedding-001"
    batch_size = 250  # verificato funzionante su Vertex
    tpm_limit = None
    output_dim = 3072  # native max

    def __init__(self) -> None:
        from google import genai

        self.client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location="us-central1",
        )

    def embed(
        self, texts: list[str], role: str
    ) -> tuple[list[list[float]], dict]:
        from google.genai.types import EmbedContentConfig

        task = "RETRIEVAL_QUERY" if role == "queries" else "RETRIEVAL_DOCUMENT"
        config = EmbedContentConfig(
            output_dimensionality=self.output_dim,
            task_type=task,
        )
        t0 = time.perf_counter()
        resp = self.client.models.embed_content(
            model=self.model, contents=texts, config=config,
        )
        latency = time.perf_counter() - t0
        vectors = [e.values for e in resp.embeddings]
        tok = sum(
            int(e.statistics.token_count) for e in resp.embeddings
            if e.statistics and e.statistics.token_count is not None
        )
        chars = None
        if resp.metadata and resp.metadata.billable_character_count is not None:
            chars = int(resp.metadata.billable_character_count)
        return vectors, {
            "input_tokens": tok,
            "input_chars": chars,
            "latency_s": latency,
        }


class GoogleGemini768Provider(GoogleProvider):
    """Gemini-embedding-001 con Matryoshka a dim 768 (stessa dim di text-005)."""
    output_dim = 768


class GoogleLegacyProvider(GoogleProvider):
    """Vertex AI text-embedding-005 (gen pre-Gemini). Native dim max = 768.

    Limite API: 20k token totali per request → batch conservativo.
    Quota Vertex default ~200k TPM → throttle lato client.
    """
    model = "text-embedding-005"
    output_dim = 768
    batch_size = 32  # 32 * ~500 tok = 16k, sotto il cap di 20k
    tpm_limit = 150_000  # conservativo vs quota Vertex ~200k TPM


class OpenAIProvider:
    name = "openai"
    model = "text-embedding-3-large"
    batch_size = 256  # conservativo (API permette 2048, ma per-request token cap)
    tpm_limit = None  # paid account: limits alti, no client throttle
    output_dim = 3072  # native max

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(
        self, texts: list[str], role: str
    ) -> tuple[list[list[float]], dict]:
        t0 = time.perf_counter()
        resp = self.client.embeddings.create(
            model=self.model, input=texts, dimensions=self.output_dim,
        )
        latency = time.perf_counter() - t0
        toks = resp.usage.prompt_tokens if resp.usage else None
        usage = {"input_tokens": toks, "latency_s": latency}
        return [d.embedding for d in resp.data], usage


class QwenProvider:
    name = "qwen"
    model = "text-embedding-v4"
    batch_size = 10  # limite Dashscope per embedding v4
    tpm_limit = None  # da scoprire sperimentalmente
    output_dim = 2048  # native max (default API = 1024)

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ["QWEN_API_KEY"],
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def embed(
        self, texts: list[str], role: str
    ) -> tuple[list[list[float]], dict]:
        t0 = time.perf_counter()
        resp = self.client.embeddings.create(
            model=self.model, input=texts, dimensions=self.output_dim,
        )
        latency = time.perf_counter() - t0
        toks = resp.usage.prompt_tokens if resp.usage else None
        usage = {"input_tokens": toks, "latency_s": latency}
        return [d.embedding for d in resp.data], usage


PROVIDERS = {
    "cohere": CohereProvider,
    "qwen": QwenProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
    "google-005": GoogleLegacyProvider,
    "google-768": GoogleGemini768Provider,
}


def load_existing(path: Path) -> tuple[set[str], list[dict]]:
    if not path.exists():
        return set(), []
    t = pq.read_table(path)
    rows = t.to_pylist()
    done = {r["id"] for r in rows}
    return done, rows


def save(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(PROVIDERS))
    ap.add_argument("--split", required=True, choices=["corpus", "queries"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="tronca N items")
    args = ap.parse_args()

    provider = PROVIDERS[args.provider]()
    items = load_split(args.split)
    if args.limit:
        items = items[: args.limit]

    model_dir = provider.model
    out_dim = getattr(provider, "output_dim", None)
    if out_dim is not None:
        model_dir = f"{provider.model}-d{out_dim}"
    out = (
        ROOT / "data" / "embeddings" / provider.name / model_dir
        / f"{args.split}.parquet"
    )
    done, rows = load_existing(out)
    todo = [it for it in items if it["id"] not in done]

    total_chars = sum(len(it["text"]) for it in todo)
    est_tokens = total_chars // 4
    n_batches = (len(todo) + provider.batch_size - 1) // provider.batch_size

    print(
        f"[info] provider={provider.name} model={provider.model} "
        f"split={args.split}"
    )
    print(
        f"[info] tot={len(items)}  fatti={len(done)}  da fare={len(todo)}  "
        f"batch_size={provider.batch_size} -> {n_batches} calls"
    )
    print(
        f"[info] ~{est_tokens:,} token stimati (char/4), "
        f"{total_chars:,} char"
    )
    pr = pricing_mod.get(provider.name, provider.model)
    if pr is not None:
        cost = pr.cost(est_tokens, total_chars)
        print(
            f"[cost] ~${cost:.4f} @ ${pr.price_usd}{pr.unit_label()} "
            f"(fonte: {pr.source})"
        )
    else:
        print("[cost] pricing non disponibile per questo modello")
    print(f"[info] output: {out}")

    if args.dry_run or not todo:
        return 0

    runs_path = out.with_name(f"{args.split}_runs.jsonl")
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    tpm = getattr(provider, "tpm_limit", None)
    tpm_target = int(tpm * 0.9) if tpm else None  # safety margin 90%
    history: deque = deque()  # (ts, tokens) nell'ultimo minuto

    pbar = tqdm(total=len(todo), desc=f"{provider.name}:{args.split}")
    start = time.time()
    cum_tokens = 0
    cum_latency = 0.0
    n_batches_done = 0
    n_waited = 0
    wait_s_total = 0.0
    i = 0
    try:
        for i in range(0, len(todo), provider.batch_size):
            batch = todo[i : i + provider.batch_size]
            batch_texts = [it["text"] for it in batch]

            if tpm_target is not None:
                est = sum(len(t) for t in batch_texts) // 4
                waited = _throttle(history, tpm_target, est)
                if waited > 0:
                    n_waited += 1
                    wait_s_total += waited

            vectors, usage = provider.embed(batch_texts, role=args.split)
            tokens_used = usage.get("input_tokens") or 0
            if tpm_target is not None:
                history.append((time.time(), tokens_used))
            cum_tokens += usage.get("input_tokens") or 0
            cum_latency += usage["latency_s"]
            n_batches_done += 1

            with runs_path.open("a") as fh:
                rec = {
                    "ts": time.time(),
                    "batch_idx": i // provider.batch_size,
                    "n_items": len(batch),
                    "input_tokens": usage.get("input_tokens"),
                    "latency_s": round(usage["latency_s"], 3),
                    "cum_input_tokens": cum_tokens,
                    "cum_latency_s": round(cum_latency, 2),
                }
                if usage.get("input_chars") is not None:
                    rec["input_chars"] = usage["input_chars"]
                fh.write(json.dumps(rec) + "\n")

            for it, v in zip(batch, vectors, strict=True):
                rows.append(
                    {
                        "id": it["id"],
                        "hash": content_hash(it["text"]),
                        "embedding": v,
                    }
                )
            save(out, rows)
            pbar.update(len(batch))
    except KeyboardInterrupt:
        print("\n[stop] interrotto da utente")
        save(out, rows)
        return 130
    except Exception as e:
        pbar.close()
        print(
            f"\n[ERR] batch a offset {i}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        save(out, rows)
        print(
            f"[save] {len(rows)} righe persistite in {out}",
            file=sys.stderr,
        )
        _report_totals(
            provider, args.split, cum_tokens, total_chars, cum_latency,
            n_batches_done, partial=True,
        )
        return 1
    finally:
        pbar.close()

    elapsed = time.time() - start
    print(f"[ok] {len(rows)} righe salvate in {out} ({elapsed:.1f}s)")
    if n_waited > 0:
        print(
            f"[throttle] {n_waited} attese per rate-limit, "
            f"{wait_s_total:.1f}s totali"
        )
    _report_totals(
        provider, args.split, cum_tokens, total_chars, cum_latency,
        n_batches_done, partial=False,
    )
    return 0


def _throttle(history: deque, tpm_target: int, est_batch_tokens: int) -> float:
    """Token-bucket sliding-window: attende se la finestra di 60s sforerebbe.

    Ritorna i secondi di attesa effettiva.
    """
    waited_total = 0.0
    while True:
        now = time.time()
        while history and history[0][0] < now - 60.0:
            history.popleft()
        used = sum(t for _, t in history)
        if used + est_batch_tokens <= tpm_target or not history:
            return waited_total
        oldest_ts, _ = history[0]
        wait = max(0.2, oldest_ts + 60.0 - now + 0.2)
        time.sleep(wait)
        waited_total += wait


def _report_totals(
    provider,
    split: str,
    tokens: int,
    chars: int,
    latency_s: float,
    n_batches: int,
    partial: bool,
) -> None:
    tag = "PARZIALE" if partial else "FINALE"
    print(f"\n[{tag}] provider={provider.name} model={provider.model} split={split}")
    print(f"  input_tokens (api) = {tokens:,}")
    print(f"  chars (local)      = {chars:,}")
    print(f"  batches            = {n_batches}")
    print(f"  latency cumul.     = {latency_s:.1f}s")
    if n_batches > 0:
        print(f"  avg latency/batch  = {latency_s / n_batches:.2f}s")
    pr = pricing_mod.get(provider.name, provider.model)
    if pr is not None and tokens > 0:
        real_cost = pr.cost(tokens, chars)
        print(f"  costo reale stim.  = ${real_cost:.4f}")
        print(f"  pricing            = ${pr.price_usd}{pr.unit_label()} ({pr.source})")


if __name__ == "__main__":
    sys.exit(main())
