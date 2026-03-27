"""Tests for the ELO-to-probability fair value model."""

import pytest
import math
import sys
sys.path.insert(0, ".")

from agent.fair_value.elo_model import EloFairValueModel
from config.settings import ArenaModel


class TestEloFairValueModel:
    def setup_method(self):
        self.model = EloFairValueModel(tau_base=150.0, sigma_elo_daily=10.0)

    def test_probabilities_sum_to_one(self):
        elos = {"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1400}
        probs = self.model.compute_probabilities(elos, days_to_expiry=30)
        assert abs(sum(probs.values()) - 1.0) < 1e-10

    def test_higher_elo_gets_higher_probability(self):
        elos = {"Anthropic": 1504, "OpenAI": 1480, "Google": 1460, "Meta": 1400}
        probs = self.model.compute_probabilities(elos, days_to_expiry=10)
        assert probs["Anthropic"] > probs["OpenAI"]
        assert probs["OpenAI"] > probs["Google"]
        assert probs["Google"] > probs["Meta"]

    def test_far_expiry_more_uniform(self):
        elos = {"Anthropic": 1504, "OpenAI": 1400}
        probs_near = self.model.compute_probabilities(elos, days_to_expiry=1)
        probs_far = self.model.compute_probabilities(elos, days_to_expiry=100)

        # Far from expiry should be more uniform (closer to 0.5 each)
        assert abs(probs_far["Anthropic"] - 0.5) < abs(probs_near["Anthropic"] - 0.5)

    def test_near_expiry_more_decisive(self):
        elos = {"Anthropic": 1504, "OpenAI": 1400}
        probs_near = self.model.compute_probabilities(elos, days_to_expiry=0.01)
        # Near expiry with big ELO gap should give strong favorite
        assert probs_near["Anthropic"] > 0.6

    def test_effective_temperature(self):
        tau_0 = self.model._effective_temperature(0)
        tau_30 = self.model._effective_temperature(30)
        assert tau_0 == pytest.approx(150.0, abs=1)
        assert tau_30 > tau_0

    def test_equal_elos_give_uniform(self):
        elos = {"A": 1400, "B": 1400, "C": 1400}
        probs = self.model.compute_probabilities(elos, days_to_expiry=10)
        for p in probs.values():
            assert p == pytest.approx(1.0 / 3, abs=0.001)

    def test_empty_input(self):
        assert self.model.compute_probabilities({}, 10) == {}

    def test_extract_company_elos(self):
        models = [
            ArenaModel("claude-opus-4", "", 1504, 6, 10000),
            ArenaModel("claude-sonnet-4", "", 1480, 6, 10000),
            ArenaModel("gpt-4o", "", 1490, 6, 10000),
            ArenaModel("gemini-2.0", "", 1460, 6, 10000),
        ]
        elos = self.model.extract_company_elos(models)
        assert elos["Anthropic"] == 1504  # Best Claude
        assert elos["OpenAI"] == 1490
        assert elos["Google"] == 1460
