"""Signal model: converts NLP outputs into calibrated likelihood ratios.

Mathematical Framework
======================

A likelihood ratio (LR) quantifies how much a news signal should shift our
belief about an event's probability. For signal s and hypothesis H (event occurs):

    LR(s) = P(s | H) / P(s | not H)

We model the log-likelihood ratio as a product of signal attributes:

    LR(s) = exp(alpha * relevance * sentiment * credibility * decay(t))

Where:
- alpha: Scaling parameter controlling signal strength (default 2.0)
- relevance: TF-IDF cosine similarity between news and market question [0, 1]
- sentiment: Combined VADER+TextBlob score [-1, 1]. Positive sentiment for
  the event yields LR > 1 (evidence for), negative yields LR < 1 (evidence against).
- credibility: Source credibility weight [0, 1]. Reuters/AP = 0.9, Reddit = 0.4.
- decay(t) = exp(-lambda * t): Exponential decay where t is hours since publication
  and lambda = ln(2) / half_life. Breaking news (t~0) has full weight; old news decays.

The exponential form ensures:
- LR is always positive (mathematical requirement)
- LR = 1 when any factor is zero (no information, no update)
- Effects compose multiplicatively in the exponent (natural for independent evidence)

LR is clamped to [0.1, 10.0] to prevent extreme updates from any single signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.settings import NewsSignal, SOURCE_CREDIBILITY


@dataclass
class SignalStrength:
    """Quantified strength of a news signal."""
    likelihood_ratio: float
    raw_strength: float  # |relevance * sentiment * credibility * decay|
    decay_factor: float
    reasoning: str


class SignalModel:
    """Converts NLP-processed news signals into calibrated likelihood ratios."""

    LR_MIN = 0.1
    LR_MAX = 10.0

    def __init__(
        self,
        alpha: float = 2.0,
        decay_half_life_hours: float = 4.0,
        source_credibility: dict[str, float] | None = None,
    ):
        self.alpha = alpha
        self.decay_half_life_hours = decay_half_life_hours
        self.decay_lambda = math.log(2) / decay_half_life_hours
        self.source_credibility = source_credibility or SOURCE_CREDIBILITY

    def compute_likelihood_ratio(self, signal: NewsSignal) -> SignalStrength:
        """Compute the likelihood ratio for a single news signal.

        Returns a SignalStrength with:
        - likelihood_ratio: LR for Bayesian updating (clamped to [0.1, 10.0])
        - raw_strength: Absolute signal strength (used for confidence computation)
        - decay_factor: How much the signal has decayed
        - reasoning: Human-readable explanation
        """
        relevance = signal.relevance_score
        sentiment = signal.sentiment_score
        credibility = signal.source_credibility
        decay = self.compute_decay(signal.hours_ago)

        # Log-likelihood ratio exponent
        exponent = self.alpha * relevance * sentiment * credibility * decay

        # LR = exp(exponent), clamped
        lr = math.exp(exponent)
        lr_clamped = max(self.LR_MIN, min(self.LR_MAX, lr))

        # Raw strength (direction-independent, for confidence)
        raw_strength = relevance * abs(sentiment) * credibility * decay

        # Human-readable reasoning
        direction = "supporting" if sentiment > 0 else "opposing" if sentiment < 0 else "neutral"
        reasoning = (
            f"{signal.source}: \"{signal.title[:60]}\" | "
            f"relevance={relevance:.2f} sentiment={sentiment:+.2f} "
            f"credibility={credibility:.2f} decay={decay:.2f} | "
            f"LR={lr_clamped:.3f} ({direction})"
        )

        return SignalStrength(
            likelihood_ratio=lr_clamped,
            raw_strength=raw_strength,
            decay_factor=decay,
            reasoning=reasoning,
        )

    def compute_decay(self, hours_since_publication: float) -> float:
        """Exponential decay factor for signal freshness.

        decay(t) = exp(-lambda * t)

        Where lambda = ln(2) / half_life, so the signal loses half its
        weight every half_life hours.

        Examples (with 4-hour half-life):
          t=0 hours: decay=1.00 (breaking news, full weight)
          t=4 hours: decay=0.50 (half weight)
          t=8 hours: decay=0.25
          t=24 hours: decay=0.016 (essentially expired)
        """
        return math.exp(-self.decay_lambda * max(0.0, hours_since_publication))

    def compute_signal_strength(self, signal: NewsSignal) -> float:
        """Scalar measure of signal informativeness, regardless of direction.

        Used for computing confidence in the Bayesian engine.
        """
        decay = self.compute_decay(signal.hours_ago)
        return (
            signal.relevance_score
            * abs(signal.sentiment_score)
            * signal.source_credibility
            * decay
        )
