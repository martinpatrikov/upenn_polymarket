"""Tests for the Kelly Criterion position sizing."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Side
from agent.scoring.kelly_criterion import KellyCriterion


def test_positive_edge_long():
    """Positive edge should yield positive position size for BUY."""
    kelly = KellyCriterion(kelly_fraction=1.0, max_position_pct=1.0)
    # q=0.60 (our estimate), p=0.50 (market price)
    # f* = (0.60 - 0.50) / (1 - 0.50) = 0.20
    result = kelly.compute_position_size(10000, 0.60, 0.50, Side.BUY)

    assert abs(result.kelly_raw - 0.20) < 0.01
    assert result.dollar_amount > 0
    assert result.side == Side.BUY


def test_negative_edge_no_position():
    """Negative edge for BUY should yield zero position."""
    kelly = KellyCriterion(kelly_fraction=1.0)
    # q=0.40, p=0.50: edge is negative for BUY
    result = kelly.compute_position_size(10000, 0.40, 0.50, Side.BUY)

    assert result.kelly_raw < 0
    assert result.dollar_amount == 0


def test_fractional_kelly():
    """Quarter Kelly should be 25% of full Kelly."""
    full_kelly = KellyCriterion(kelly_fraction=1.0, max_position_pct=1.0)
    quarter_kelly = KellyCriterion(kelly_fraction=0.25, max_position_pct=1.0)

    full = full_kelly.compute_position_size(10000, 0.70, 0.50, Side.BUY)
    quarter = quarter_kelly.compute_position_size(10000, 0.70, 0.50, Side.BUY)

    assert abs(quarter.kelly_scaled - full.kelly_raw * 0.25) < 0.01


def test_max_position_cap():
    """Position should be capped at max_position_pct."""
    kelly = KellyCriterion(kelly_fraction=1.0, max_position_pct=0.05)
    # Very large edge -> huge Kelly, but capped at 5%
    result = kelly.compute_position_size(10000, 0.95, 0.50, Side.BUY)

    assert result.kelly_scaled <= 0.05
    assert result.dollar_amount <= 10000 * 0.05 + 0.01  # small float tolerance


def test_sell_side_kelly():
    """SELL side Kelly should work correctly."""
    kelly = KellyCriterion(kelly_fraction=1.0, max_position_pct=1.0)
    # q=0.30 (event less likely), p=0.50 (market price)
    # SELL f* = (p - q) / p = (0.50 - 0.30) / 0.50 = 0.40
    result = kelly.compute_position_size(10000, 0.30, 0.50, Side.SELL)

    assert abs(result.kelly_raw - 0.40) < 0.01
    assert result.dollar_amount > 0
    assert result.side == Side.SELL


def test_no_edge_no_trade():
    """When estimated prob equals market price, Kelly should be zero."""
    kelly = KellyCriterion(kelly_fraction=0.25)
    result = kelly.compute_position_size(10000, 0.50, 0.50, Side.BUY)

    assert abs(result.kelly_raw) < 0.02  # approximately zero
    assert result.dollar_amount == 0 or result.dollar_amount < 1


def test_contract_calculation():
    """Number of contracts should be dollar_amount / price."""
    kelly = KellyCriterion(kelly_fraction=0.25, max_position_pct=0.10)
    result = kelly.compute_position_size(10000, 0.70, 0.50, Side.BUY)

    if result.dollar_amount > 0:
        expected_contracts = result.dollar_amount / 0.50
        assert abs(result.num_contracts - expected_contracts) < 0.01


if __name__ == "__main__":
    test_positive_edge_long()
    test_negative_edge_no_position()
    test_fractional_kelly()
    test_max_position_cap()
    test_sell_side_kelly()
    test_no_edge_no_trade()
    test_contract_calculation()
    print("All Kelly criterion tests passed!")
