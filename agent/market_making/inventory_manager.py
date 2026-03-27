"""Multi-outcome inventory tracking for the market making agent."""

import logging
from config.settings import InventoryState, Side

logger = logging.getLogger(__name__)


class InventoryManager:
    """Tracks inventory across all outcomes and enforces limits."""

    def __init__(
        self,
        max_inventory_per_outcome: float = 100.0,
        max_total_inventory: float = 300.0,
    ):
        self.max_per_outcome = max_inventory_per_outcome
        self.max_total = max_total_inventory
        self._positions: dict[str, float] = {}  # outcome -> signed quantity
        self._avg_prices: dict[str, float] = {}  # outcome -> avg entry price
        self._realized_pnl: float = 0.0

    def update(self, outcome: str, side: Side, quantity: float, price: float) -> None:
        """Record a fill (buy or sell)."""
        current = self._positions.get(outcome, 0.0)
        current_avg = self._avg_prices.get(outcome, 0.0)

        if side == Side.BUY:
            new_qty = current + quantity
            if current >= 0:
                # Adding to long: update avg price
                total_cost = current * current_avg + quantity * price
                self._avg_prices[outcome] = total_cost / new_qty if new_qty > 0 else 0
            else:
                # Reducing short: realize P&L
                closed = min(quantity, abs(current))
                self._realized_pnl += closed * (current_avg - price)
                if new_qty > 0:
                    self._avg_prices[outcome] = price
            self._positions[outcome] = new_qty

        elif side == Side.SELL:
            new_qty = current - quantity
            if current <= 0:
                # Adding to short: update avg price
                total_cost = abs(current) * current_avg + quantity * price
                self._avg_prices[outcome] = total_cost / abs(new_qty) if new_qty != 0 else 0
            else:
                # Reducing long: realize P&L
                closed = min(quantity, current)
                self._realized_pnl += closed * (price - current_avg)
                if new_qty < 0:
                    self._avg_prices[outcome] = price
            self._positions[outcome] = new_qty

    def get_inventory(self, outcome: str) -> float:
        """Get current inventory for an outcome (positive = long, negative = short)."""
        return self._positions.get(outcome, 0.0)

    def get_all_inventories(self) -> dict[str, float]:
        """Get all inventories."""
        return dict(self._positions)

    def get_state(self, fair_values: dict[str, float]) -> InventoryState:
        """Get full inventory state with unrealized P&L."""
        unrealized = 0.0
        for outcome, qty in self._positions.items():
            avg = self._avg_prices.get(outcome, 0.0)
            fv = fair_values.get(outcome, 0.0)
            if qty > 0:
                unrealized += qty * (fv - avg)
            elif qty < 0:
                unrealized += abs(qty) * (avg - fv)

        return InventoryState(
            positions=dict(self._positions),
            avg_prices=dict(self._avg_prices),
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized,
        )

    def should_reduce(self, outcome: str, side: Side) -> bool:
        """Check if adding to this side would breach inventory limits."""
        current = self._positions.get(outcome, 0.0)
        if side == Side.BUY and current >= self.max_per_outcome:
            return True
        if side == Side.SELL and current <= -self.max_per_outcome:
            return True
        total = sum(abs(q) for q in self._positions.values())
        if total >= self.max_total:
            return True
        return False

    def reset(self) -> None:
        """Reset all positions."""
        self._positions.clear()
        self._avg_prices.clear()
        self._realized_pnl = 0.0
