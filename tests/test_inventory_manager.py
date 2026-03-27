"""Tests for multi-outcome inventory management."""

import pytest
import sys
sys.path.insert(0, ".")

from agent.market_making.inventory_manager import InventoryManager
from config.settings import Side


class TestInventoryManager:
    def setup_method(self):
        self.mgr = InventoryManager(
            max_inventory_per_outcome=100.0,
            max_total_inventory=300.0,
        )

    def test_initial_inventory_zero(self):
        assert self.mgr.get_inventory("Anthropic") == 0.0

    def test_buy_increases_inventory(self):
        self.mgr.update("Anthropic", Side.BUY, 50, 0.45)
        assert self.mgr.get_inventory("Anthropic") == 50.0

    def test_sell_decreases_inventory(self):
        self.mgr.update("Anthropic", Side.BUY, 50, 0.45)
        self.mgr.update("Anthropic", Side.SELL, 20, 0.50)
        assert self.mgr.get_inventory("Anthropic") == 30.0

    def test_realized_pnl_on_sell(self):
        self.mgr.update("Anthropic", Side.BUY, 50, 0.40)
        self.mgr.update("Anthropic", Side.SELL, 50, 0.50)
        state = self.mgr.get_state({"Anthropic": 0.50})
        # Bought at 0.40, sold at 0.50 -> profit = 50 * 0.10 = 5.0
        assert state.realized_pnl == pytest.approx(5.0, abs=0.01)

    def test_unrealized_pnl(self):
        self.mgr.update("Anthropic", Side.BUY, 100, 0.40)
        state = self.mgr.get_state({"Anthropic": 0.50})
        # Holding 100 at avg 0.40, market 0.50 -> unrealized = 100 * 0.10 = 10.0
        assert state.unrealized_pnl == pytest.approx(10.0, abs=0.01)

    def test_should_reduce_at_limit(self):
        self.mgr.update("Anthropic", Side.BUY, 100, 0.45)
        assert self.mgr.should_reduce("Anthropic", Side.BUY) is True
        assert self.mgr.should_reduce("Anthropic", Side.SELL) is False

    def test_multiple_outcomes(self):
        self.mgr.update("Anthropic", Side.BUY, 50, 0.45)
        self.mgr.update("OpenAI", Side.BUY, 30, 0.30)
        assert self.mgr.get_inventory("Anthropic") == 50.0
        assert self.mgr.get_inventory("OpenAI") == 30.0

    def test_reset(self):
        self.mgr.update("Anthropic", Side.BUY, 50, 0.45)
        self.mgr.reset()
        assert self.mgr.get_inventory("Anthropic") == 0.0
