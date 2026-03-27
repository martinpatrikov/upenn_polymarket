"""Adjusts ELO-based fair value probabilities with real-time news/Twitter signals.

Uses likelihood ratios in odds space to shift probabilities, then renormalizes.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone

from config.settings import NewsSignal, COMPANY_MODEL_PREFIXES

logger = logging.getLogger(__name__)

# Keywords that map signals to specific companies
COMPANY_KEYWORDS: dict[str, list[str]] = {
    "Anthropic": ["anthropic", "claude", "sonnet", "haiku", "opus"],
    "OpenAI": ["openai", "gpt", "chatgpt", "o1", "o3", "o4", "dall-e"],
    "Google": ["google", "deepmind", "gemini", "gemma", "bard"],
    "Meta": ["meta", "llama", "facebook"],
    "xAI": ["xai", "grok", "elon"],
    "Mistral": ["mistral", "codestral", "mixtral"],
    "DeepSeek": ["deepseek"],
}


class SignalAdjuster:
    """Adjusts fair value probabilities based on news/Twitter signals."""

    def __init__(
        self,
        alpha: float = 2.0,
        decay_half_life_hours: float = 4.0,
        max_shift: float = 0.15,
    ):
        self.alpha = alpha
        self.decay_lambda = math.log(2) / decay_half_life_hours
        self.max_shift = max_shift

    def adjust_probabilities(
        self,
        base_probs: dict[str, float],
        signals: list[NewsSignal],
    ) -> dict[str, float]:
        """Adjust probabilities based on signals.

        For each signal:
        1. Determine which company it affects
        2. Compute a likelihood ratio
        3. Apply in odds space
        4. Renormalize

        Args:
            base_probs: {company: probability} from ELO model.
            signals: News/Twitter signals with relevance and sentiment.

        Returns:
            Adjusted {company: probability} summing to 1.0.
        """
        if not signals or not base_probs:
            return base_probs

        adjusted = dict(base_probs)

        for signal in signals:
            company = self._identify_company(signal)
            if company is None or company not in adjusted:
                continue

            lr = self._compute_likelihood_ratio(signal)
            if abs(lr - 1.0) < 0.001:
                continue  # No meaningful shift

            # Apply LR in odds space
            old_prob = adjusted[company]
            if old_prob <= 0.001 or old_prob >= 0.999:
                continue

            old_odds = old_prob / (1.0 - old_prob)
            new_odds = old_odds * lr
            new_prob = new_odds / (1.0 + new_odds)

            # Cap the shift
            shift = new_prob - old_prob
            if abs(shift) > self.max_shift:
                new_prob = old_prob + (self.max_shift if shift > 0 else -self.max_shift)

            adjusted[company] = max(0.001, min(0.999, new_prob))

        # Renormalize to sum to 1
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {c: p / total for c, p in adjusted.items()}

        return adjusted

    def _compute_likelihood_ratio(self, signal: NewsSignal) -> float:
        """Compute likelihood ratio from a signal.

        LR = exp(alpha * relevance * sentiment * credibility * decay)
        """
        now = datetime.now(timezone.utc)
        if signal.published_at.tzinfo is None:
            age_hours = 0.0
        else:
            age_hours = (now - signal.published_at).total_seconds() / 3600.0

        decay = math.exp(-self.decay_lambda * max(0, age_hours))
        relevance = signal.relevance_score
        sentiment = signal.sentiment_score
        credibility = signal.source_credibility

        exponent = self.alpha * relevance * sentiment * credibility * decay

        # Clamp to prevent extreme values
        exponent = max(-2.3, min(2.3, exponent))  # LR in [0.1, 10.0]

        return math.exp(exponent)

    def _identify_company(self, signal: NewsSignal) -> str | None:
        """Determine which company a signal is about."""
        text = (signal.title + " " + signal.summary).lower()

        best_company = None
        best_score = 0

        for company, keywords in COMPANY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_company = company

        return best_company
