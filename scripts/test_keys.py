"""Smoke test per le API key dei provider di embedding nel .env."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SAMPLE = "The quick brown fox jumps over the lazy dog."


def test_openai() -> tuple[bool, str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "text-embedding-3-large"
    resp = client.embeddings.create(model=model, input=SAMPLE)
    dim = len(resp.data[0].embedding)
    return True, f"{model} dim={dim}"


def test_cohere() -> tuple[bool, str]:
    import cohere

    client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    model = "embed-v4.0"
    resp = client.embed(
        texts=[SAMPLE],
        model=model,
        input_type="search_document",
        embedding_types=["float"],
    )
    dim = len(resp.embeddings.float_[0])
    return True, f"{model} dim={dim}"


def test_vertex() -> tuple[bool, str]:
    from google import genai

    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location="us-central1",
    )
    model_name = "gemini-embedding-001"
    resp = client.models.embed_content(model=model_name, contents=SAMPLE)
    dim = len(resp.embeddings[0].values)
    return True, f"{model_name} dim={dim}"


def test_qwen() -> tuple[bool, str]:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["QWEN_API_KEY"],
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    model = "text-embedding-v4"
    resp = client.embeddings.create(model=model, input=SAMPLE)
    dim = len(resp.data[0].embedding)
    return True, f"{model} dim={dim}"


TESTS = [
    ("OpenAI", test_openai),
    ("Cohere", test_cohere),
    ("Vertex AI", test_vertex),
    ("Qwen (Dashscope)", test_qwen),
]


def main() -> int:
    failures = 0
    for name, fn in TESTS:
        try:
            ok, info = fn()
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {name:18s} {info}")
            if not ok:
                failures += 1
        except Exception as e:
            failures += 1
            print(f"[FAIL] {name:18s} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
