"""Volatility estimation for the Avellaneda-Stoikov model.

Estimates per-outcome price volatility from ELO uncertainty,
translated into probability space.

Key formula:
    sigma_p = p * (1 - p) / tau_eff * sigma_elo
"""

import math
import logging

logger = logging.getLogger(__name__)


class VolatilityEstimator:
    """Estimates per-outcome volatility for A-S quoting."""

    def __init__(
        self,
        sigma_elo_daily: float = 10.0,
        min_sigma: float = 0.01,
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

        Uses ELO-implied volatility translated to probability space:
            sigma_p = p * (1 - p) / tau_eff * sigma_elo

        This naturally gives:
        - Higher volatility for outcomes near 50%
        - Lower volatility for extreme outcomes (near 0% or 100%)

        Args:
            probabilities: {outcome: probability}
            tau_eff: Effective temperature from ELO model.

        Returns:
            {outcome: sigma} clamped to [min_sigma, max_sigma]
        """
        sigmas = {}
        for outcome, prob in probabilities.items():
            # dp/dE for softmax: p * (1 - p) / tau
            dp_dE = prob * (1.0 - prob) / tau_eff
            sigma_p = dp_dE * self.sigma_elo_daily

            # Clamp
            sigma_p = max(self.min_sigma, min(self.max_sigma, sigma_p))
            sigmas[outcome] = sigma_p

        return sigmas

    def estimate_from_price_history(
        self,
        price_history: list[float],
    ) -> float:
        """Estimate volatility from a series of prices (optional).

        Uses standard deviation of log returns.
        """
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
