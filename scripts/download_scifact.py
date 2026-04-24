"""Scarica il dataset SciFact di BEIR da public.ukp.informatik.tu-darmstadt.de."""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "scifact"
    if target.exists():
        print(f"[skip] {target} esiste già")
        return 0

    print(f"[get] {URL}")
    with urllib.request.urlopen(URL) as r:
        data = r.read()
    print(f"[get] scaricati {len(data) / 1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(OUT_DIR)
    print(f"[ok] estratto in {target}")

    for p in sorted(target.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(target)}: {p.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
