"""Bayesian confidence estimation engine.

Mathematical Framework
======================

1. PRIOR: Market-implied probability from the orderbook midpoint.
   If midpoint = 0.65, then:
     P(H) = 0.65
     prior_odds = P(H) / P(not H) = 0.65 / 0.35 = 1.857

2. SEQUENTIAL BAYESIAN UPDATING: For each signal s_i with likelihood ratio LR_i:
     posterior_odds = prior_odds * LR_1 * LR_2 * ... * LR_n
     posterior_probability = posterior_odds / (1 + posterior_odds)

   This is mathematically equivalent to applying Bayes' theorem repeatedly,
   assuming signals are conditionally independent given the hypothesis.

3. EDGE CALCULATION:
     edge = posterior_probability - market_probability
   Positive edge on YES: agent believes event is more likely than market implies.
   Negative edge: agent believes event is less likely.

4. CONFIDENCE IN THE ESTIMATE (distinct from the probability):
   Measures how much evidence supports our divergence from the market.
     information_weight = sum(signal_strength_i for all signals)
     confidence = 1 - exp(-beta * information_weight)
   This gives:
     - 0 confidence with no signals (don't trade without evidence)
     - Asymptotically approaches 1 with many strong signals
     - beta controls how quickly confidence grows (default 1.5)

5. EFFECTIVE EDGE:
     effective_edge = edge * confidence
   Ensures we do not trade on thin evidence, even if the edge looks large.
   A 20% edge with 10% confidence is only 2% effective edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.settings import NewsSignal, PosteriorEstimate
from agent.scoring.signal_model import SignalModel


class BayesianEngine:
    """Core probability estimation engine using Bayesian updating."""

    # Clamp market prices away from 0 and 1 to avoid infinite odds
    PRICE_FLOOR = 0.01
    PRICE_CEILING = 0.99

    def __init__(
        self,
        signal_model: SignalModel,
        min_edge: float = 0.03,
        min_confidence: float = 0.20,
        confidence_beta: float = 1.5,
    ):
        self.signal_model = signal_model
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.confidence_beta = confidence_beta

    def compute_posterior(
        self, market_price: float, signals: list[NewsSignal]
    ) -> PosteriorEstimate:
        """Compute posterior probability by Bayesian updating with news signals.

        Args:
            market_price: Current market-implied probability (midpoint price).
            signals: List of processed news signals with relevance and sentiment.

        Returns:
            PosteriorEstimate with prior, posterior, edge, confidence, and reasoning.
        """
        # Clamp market price
        market_price = max(self.PRICE_FLOOR, min(self.PRICE_CEILING, market_price))

        # Step 1: Prior odds from market price
        prior_odds = self._compute_prior_odds(market_price)

        # Step 2: Compute likelihood ratios and update odds sequentially
        likelihood_ratios: list[float] = []
        reasoning: list[str] = []
        total_signal_strength = 0.0

        current_odds = prior_odds
        for signal in signals:
            result = self.signal_model.compute_likelihood_ratio(signal)
            lr = result.likelihood_ratio
            likelihood_ratios.append(lr)
            reasoning.append(result.reasoning)
            total_signal_strength += result.raw_strength

            # Sequential update
            current_odds *= lr

        # Step 3: Convert back to probability
        posterior_prob = current_odds / (1 + current_odds)

        # Step 4: Edge over market
        edge = posterior_prob - market_price

        # Step 5: Confidence from information weight
        confidence = 1 - math.exp(-self.confidence_beta * total_signal_strength)

        # Step 6: Effective edge
        effective_edge = edge * confidence

        return PosteriorEstimate(
            prior_probability=market_price,
            posterior_probability=posterior_prob,
            edge=edge,
            confidence=confidence,
            effective_edge=effective_edge,
            likelihood_ratios=likelihood_ratios,
            signal_count=len(signals),
            reasoning=reasoning,
        )

    def should_trade(self, estimate: PosteriorEstimate) -> bool:
        """Determine if the estimated edge is sufficient to trade.

        Requires both:
        - |effective_edge| >= min_edge (enough edge after confidence scaling)
        - confidence >= min_confidence (enough evidence to act)
        """
        return (
            abs(estimate.effective_edge) >= self.min_edge
            and estimate.confidence >= self.min_confidence
        )

    def _compute_prior_odds(self, market_price: float) -> float:
        """Convert market probability to odds form: p / (1 - p)."""
        return market_price / (1 - market_price)
