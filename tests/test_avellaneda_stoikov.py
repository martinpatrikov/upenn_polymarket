"""Tests for the Avellaneda-Stoikov market making model."""

import pytest
import math
import sys
sys.path.insert(0, ".")

from agent.market_making.avellaneda_stoikov import AvellanedaStoikov


class TestAvellanedaStoikov:
    def setup_method(self):
        self.engine = AvellanedaStoikov(
            gamma=0.1, kappa=1.5, base_order_size=10.0, max_inventory=100.0
        )

    def test_reservation_price_zero_inventory(self):
        """With no inventory, reservation price equals fair value."""
        r = self.engine.compute_reservation_price(0.50, 0.0, 0.05, 10)
        assert r == pytest.approx(0.50, abs=0.001)

    def test_reservation_price_long_inventory(self):
        """Long inventory pushes reservation price down (eager to sell)."""
        r_long = self.engine.compute_reservation_price(0.50, 50.0, 0.05, 10)
        r_zero = self.engine.compute_reservation_price(0.50, 0.0, 0.05, 10)
        assert r_long < r_zero

    def test_reservation_price_short_inventory(self):
        """Short inventory pushes reservation price up (eager to buy)."""
        r_short = self.engine.compute_reservation_price(0.50, -50.0, 0.05, 10)
        r_zero = self.engine.compute_reservation_price(0.50, 0.0, 0.05, 10)
        assert r_short > r_zero

    def test_optimal_spread_positive(self):
        delta = self.engine.compute_optimal_spread(0.05, 10)
        assert delta > 0

    def test_spread_increases_with_volatility(self):
        delta_low = self.engine.compute_optimal_spread(0.02, 10)
        delta_high = self.engine.compute_optimal_spread(0.10, 10)
        assert delta_high > delta_low

    def test_spread_increases_with_time(self):
        delta_short = self.engine.compute_optimal_spread(0.05, 1)
        delta_long = self.engine.compute_optimal_spread(0.05, 30)
        assert delta_long > delta_short

    def test_quotes_bid_less_than_ask(self):
        quotes = self.engine.generate_quotes(
            fair_values={"A": 0.50, "B": 0.30},
            inventories={"A": 0.0, "B": 0.0},
            sigmas={"A": 0.05, "B": 0.05},
            time_to_expiry=10,
            token_ids={"A": "tok_a", "B": "tok_b"},
        )
        for q in quotes:
            assert q.bid_price < q.ask_price
            assert q.bid_price >= 0.01
            assert q.ask_price <= 0.99

    def test_quotes_clamped_to_bounds(self):
        """Extreme values should be clamped."""
        quotes = self.engine.generate_quotes(
            fair_values={"A": 0.01},
            inventories={"A": 0.0},
            sigmas={"A": 0.15},
            time_to_expiry=30,
            token_ids={"A": "tok_a"},
        )
        assert quotes[0].bid_price >= 0.01
        assert quotes[0].ask_price <= 0.99

    def test_inventory_skew_reduces_bid_size_when_long(self):
        q_long = self.engine._adjust_size(50.0, "bid")
        q_zero = self.engine._adjust_size(0.0, "bid")
        assert q_long < q_zero

    def test_inventory_skew_increases_ask_size_when_long(self):
        q_long = self.engine._adjust_size(50.0, "ask")
        q_zero = self.engine._adjust_size(0.0, "ask")
        assert q_long > q_zero
