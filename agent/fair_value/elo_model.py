"""ELO-to-probability conversion for AI model prediction markets.

Converts Chatbot Arena ELO scores into outcome probabilities for each company
using a temperature-scaled softmax with time adjustment.

Key formula:
    tau_eff(t) = tau_base + sigma_elo * sqrt(T - t)
    p_c = exp(E_c / tau_eff) / sum_k(exp(E_k / tau_eff))
"""

import math
import logging
from typing import Optional

from config.settings import ArenaModel, COMPANY_MODEL_PREFIXES

logger = logging.getLogger(__name__)


class EloFairValueModel:
    """Converts Arena ELO scores to outcome probabilities."""

    def __init__(
        self,
        tau_base: float = 150.0,
        sigma_elo_daily: float = 10.0,
        company_prefixes: Optional[dict[str, list[str]]] = None,
    ):
        self.tau_base = tau_base
        self.sigma_elo_daily = sigma_elo_daily
        self.company_prefixes = company_prefixes or COMPANY_MODEL_PREFIXES

    def compute_probabilities(
        self,
        company_elos: dict[str, float],
        days_to_expiry: float,
    ) -> dict[str, float]:
        """Compute outcome probabilities from company ELO scores.

        Args:
            company_elos: {company_name: best_elo_score}
            days_to_expiry: Days until market resolution.

        Returns:
            {company_name: probability} summing to 1.0
        """
        if not company_elos:
            return {}

        tau_eff = self._effective_temperature(days_to_expiry)

        # Temperature-scaled softmax
        # Subtract max for numerical stability
        max_elo = max(company_elos.values())
        exp_scores = {}
        for company, elo in company_elos.items():
            exp_scores[company] = math.exp((elo - max_elo) / tau_eff)

        total = sum(exp_scores.values())
        if total == 0:
            # Uniform fallback
            n = len(company_elos)
            return {c: 1.0 / n for c in company_elos}

        probs = {c: v / total for c, v in exp_scores.items()}

        logger.debug(
            f"ELO->probs (tau_eff={tau_eff:.1f}, days={days_to_expiry:.1f}): "
            + ", ".join(f"{c}={p:.3f}" for c, p in sorted(probs.items(), key=lambda x: -x[1]))
        )

        return probs

    def _effective_temperature(self, days_to_expiry: float) -> float:
        """Compute effective temperature.

        Far from expiry: high temperature -> probabilities pulled toward uniform.
        Close to expiry: base temperature -> ELO scores dominate.
        """
        days = max(0.0, days_to_expiry)
        return self.tau_base + self.sigma_elo_daily * math.sqrt(days)

    def extract_company_elos(
        self,
        models: list[ArenaModel],
        target_companies: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Extract best ELO per company from model list.

        Args:
            models: List of ArenaModel from the scraper.
            target_companies: If provided, only include these companies.

        Returns:
            {company_name: best_elo_score}
        """
        company_best: dict[str, float] = {}

        for model in models:
            company = self._classify_company(model.model_name)
            if company is None:
                continue
            if target_companies and company not in target_companies:
                continue
            if company not in company_best or model.elo_score > company_best[company]:
                company_best[company] = model.elo_score

        return company_best

    def _classify_company(self, model_name: str) -> Optional[str]:
        """Map model name to company via prefix matching."""
        name_lower = model_name.lower()
        for company, prefixes in self.company_prefixes.items():
            for prefix in prefixes:
                if name_lower.startswith(prefix):
                    return company
        return None
