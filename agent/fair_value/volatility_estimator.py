"""Volatility estimation for the Avellaneda-Stoikov model.

Estimates per-outcome price volatility combining ELO-implied volatility
with empirical prediction market volatility floors.

In prediction markets, prices move from:
1. ELO score changes (captured by dp/dE * sigma_elo)
2. News/sentiment shifts (not captured by ELO)
3. Liquidity events (not captured by ELO)

We use max(elo_implied, empirical_floor) to ensure the A-S model
quotes realistically wide spreads and reacts to inventory.
"""

import math
import logging

logger = logging.getLogger(__name__)

# Empirical daily volatility for prediction market outcomes.
# Based on observed Polymarket price movements:
# - Outcomes near 50%: ~5-8% daily moves
# - Outcomes near 10%: ~2-4% daily moves
# - Outcomes near 1%:  ~0.5-1% daily moves
# These are floors — actual volatility can be higher.
EMPIRICAL_SIGMA_FLOOR = 0.03


class VolatilityEstimator:
    """Estimates per-outcome volatility for A-S quoting."""

    def __init__(
        self,
        sigma_elo_daily: float = 10.0,
        min_sigma: float = 0.02,
        max_sigma: float = 0.20,
    ):
        self.sigma_elo_daily = sigma_elo_daily
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def estimate(
        self,
        probabilities: dict[str, float],
        tau_eff: float,
    ) -> dict[str, float]:
        """Estimate volatility for each outcome.

        Combines:
        1. ELO-implied: sigma = p*(1-p)/tau_eff * sigma_elo
        2. Empirical floor: scales with p*(1-p) (most volatile near 50%)

        Uses the max of both to capture all sources of price movement.
        """
        sigmas = {}
        for outcome, prob in probabilities.items():
            # ELO-implied volatility
            dp_dE = prob * (1.0 - prob) / tau_eff
            sigma_elo = dp_dE * self.sigma_elo_daily

            # Empirical floor: prediction markets have baseline volatility
            # that scales with p*(1-p) — outcomes near 50% are most volatile
            sigma_empirical = EMPIRICAL_SIGMA_FLOOR + 0.10 * prob * (1.0 - prob)

            # Use the higher estimate
            sigma = max(sigma_elo, sigma_empirical)

            # Clamp
            sigma = max(self.min_sigma, min(self.max_sigma, sigma))
            sigmas[outcome] = sigma

        return sigmas

    def estimate_from_price_history(
        self,
        price_history: list[float],
    ) -> float:
        """Estimate volatility from a series of prices."""
        if len(price_history) < 3:
            return self.min_sigma

        log_returns = []
        for i in range(1, len(price_history)):
            if price_history[i] > 0 and price_history[i - 1] > 0:
                log_returns.append(math.log(price_history[i] / price_history[i - 1]))

        if not log_returns:
            return self.min_sigma

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        sigma = math.sqrt(variance)

        return max(self.min_sigma, min(self.max_sigma, sigma))
