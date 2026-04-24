"""Pricing indicativo dei provider di embedding.

ATTENZIONE: valori "best-effort" presi a memoria dalle pagine di pricing
pubbliche intorno ad aprile 2026. Vanno *ri-verificati* dalle docs ufficiali
prima di citarli in articoli/pubblicazioni. Per aggiornarli, modifica il dict
`PRICING` qui sotto.

Unità:
    - `per_1m_tokens`: USD per 1M token di input
    - `per_1m_chars`:  USD per 1M caratteri di input (usato da Vertex AI)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    provider: str
    model: str
    price_usd: float
    unit: str  # "per_1m_tokens" | "per_1m_chars"
    source: str

    def cost(self, tokens_est: int, chars: int) -> float:
        if self.unit == "per_1m_tokens":
            return tokens_est * self.price_usd / 1_000_000
        if self.unit == "per_1m_chars":
            return chars * self.price_usd / 1_000_000
        raise ValueError(f"unit sconosciuta: {self.unit}")

    def unit_label(self) -> str:
        return "/1M tok" if self.unit == "per_1m_tokens" else "/1M char"


PRICING: dict[tuple[str, str], Pricing] = {
    ("openai", "text-embedding-3-large"): Pricing(
        "openai", "text-embedding-3-large",
        price_usd=0.13, unit="per_1m_tokens",
        source="openai.com/api/pricing (confermato 2026-04)",
    ),
    ("openai", "text-embedding-3-small"): Pricing(
        "openai", "text-embedding-3-small",
        price_usd=0.02, unit="per_1m_tokens",
        source="openai.com/api/pricing (confermato 2026-04)",
    ),
    ("cohere", "embed-v4.0"): Pricing(
        "cohere", "embed-v4.0",
        price_usd=0.12, unit="per_1m_tokens",
        source="pricepertoken.com/provider/cohere (verificato 2026-04)",
    ),
    ("qwen", "text-embedding-v4"): Pricing(
        "qwen", "text-embedding-v4",
        price_usd=0.07, unit="per_1m_tokens",
        # free tier: 1M token / 90gg dall'attivazione Model Studio
        source="alibabacloud.com/help/en/model-studio/billing-for-text-embedding (verificato 2026-04)",
    ),
    ("google", "gemini-embedding-001"): Pricing(
        "google", "gemini-embedding-001",
        price_usd=0.15, unit="per_1m_tokens",
        # batch mode: $0.075/1M tokens (non usata qui)
        source="ai.google.dev/gemini-api/docs/pricing (verificato 2026-04)",
    ),
    ("google", "text-embedding-005"): Pricing(
        "google", "text-embedding-005",
        price_usd=0.025, unit="per_1m_chars",
        source="cloud.google.com/vertex-ai/pricing (verificare)",
    ),
}


def get(provider: str, model: str) -> Pricing | None:
    return PRICING.get((provider, model))
