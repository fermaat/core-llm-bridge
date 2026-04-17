"""
LLM cost tracking for core-llm-bridge.

Tracks token usage and estimated cost per call and cumulatively.
Auto-integrated with BridgeEngine — every chat() call populates
response.cost_usd and records an entry in the cost_tracker singleton.

Pricing is hardcoded in DEFAULT_PRICING. Override per model via
cost_tracker.set_pricing("my-model", input_per_1m=1.0, output_per_1m=3.0)
or replace the full dict via cost_tracker.pricing = {...}.

Any model not listed defaults to $0.00 (covers all Ollama models).

Usage:
    from core_llm_bridge.cost_tracker import cost_tracker

    response = engine.chat("summarise this")
    print(f"This call: ${response.cost_usd:.6f}")

    cost_tracker.report()   # cumulative log summary
    data = cost_tracker.to_dict()
    cost_tracker.reset()
"""

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ── Pricing ───────────────────────────────────────────────────────────────────


@dataclass
class ModelPricing:
    """USD cost per 1 million tokens."""

    input_per_1m: float
    output_per_1m: float


DEFAULT_PRICING: dict[str, ModelPricing] = {
    # Anthropic
    "claude-opus-4-7": ModelPricing(input_per_1m=15.0, output_per_1m=75.0),
    "claude-sonnet-4-6": ModelPricing(input_per_1m=3.0, output_per_1m=15.0),
    "claude-haiku-4-5-20251001": ModelPricing(input_per_1m=0.8, output_per_1m=4.0),
    # OpenAI
    "gpt-4o": ModelPricing(input_per_1m=2.5, output_per_1m=10.0),
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, output_per_1m=0.6),
    "gpt-4-turbo": ModelPricing(input_per_1m=10.0, output_per_1m=30.0),
    # Any model not listed (e.g. all Ollama models) → $0.00
}


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class CostEntry:
    """Record of a single tracked LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "label": self.label,
        }


# ── Tracker ───────────────────────────────────────────────────────────────────


class CostTracker:
    """
    Accumulates LLM call costs and reports totals.

    Use the module-level `cost_tracker` singleton rather than instantiating
    directly. BridgeEngine populates it automatically on every call.
    """

    def __init__(self, pricing: dict[str, ModelPricing] | None = None) -> None:
        self.pricing: dict[str, ModelPricing] = pricing or dict(DEFAULT_PRICING)
        self._entries: list[CostEntry] = []

    def set_pricing(self, model: str, input_per_1m: float, output_per_1m: float) -> None:
        """Override pricing for a specific model."""
        self.pricing[model] = ModelPricing(input_per_1m=input_per_1m, output_per_1m=output_per_1m)

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Return estimated cost in USD without recording an entry."""
        p = self.pricing.get(model, ModelPricing(0.0, 0.0))
        return (input_tokens * p.input_per_1m + output_tokens * p.output_per_1m) / 1_000_000

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        label: str | None = None,
    ) -> CostEntry:
        """Record a call and return its CostEntry."""
        cost = self.estimate(model, input_tokens, output_tokens)
        entry = CostEntry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            label=label,
        )
        self._entries.append(entry)
        return entry

    def total(self) -> float:
        """Total accumulated cost in USD."""
        return sum(e.cost_usd for e in self._entries)

    def reset(self) -> None:
        """Clear all recorded entries."""
        self._entries.clear()

    def report(self) -> None:
        """Log a summary of all recorded calls."""
        if not self._entries:
            logger.info("[cost] no calls recorded")
            return

        lines = [f"[cost] total: ${self.total():.6f}  |  {len(self._entries)} call(s)"]
        for e in self._entries:
            label = f"  [{e.label}]" if e.label else ""
            lines.append(
                f"  {e.model:<40} in={e.input_tokens:<6} out={e.output_tokens:<6}"
                f"  ${e.cost_usd:.6f}{label}"
            )
        logger.info("\n" + "\n".join(lines))

    def to_dict(self) -> dict[str, Any]:
        """Export all entries and totals as a dict."""
        return {
            "total_usd": round(self.total(), 8),
            "calls": len(self._entries),
            "entries": [e.to_dict() for e in self._entries],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ── Singleton ─────────────────────────────────────────────────────────────────

cost_tracker = CostTracker()

__all__ = ["cost_tracker", "CostTracker", "CostEntry", "ModelPricing", "DEFAULT_PRICING"]
