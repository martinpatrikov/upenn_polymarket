"""Avellaneda-Stoikov market making model adapted for prediction markets.

Core formulas:
    Reservation price: r = p_hat - q * gamma * sigma^2 * (T - t)
    Optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)
    Bid: r - delta/2
    Ask: r + delta/2
"""

import math
import logging
from datetime import datetime

from config.settings import ASQuote

logger = logging.getLogger(__name__)

PRICE_FLOOR = 0.01
PRICE_CEILING = 0.99


class AvellanedaStoikov:
    """Avellaneda-Stoikov market making model for prediction markets."""

    def __init__(
        self,
        gamma: float = 0.1,
        kappa: float = 1.5,
        base_order_size: float = 10.0,
        max_inventory: float = 100.0,
    ):
        self.gamma = gamma
        self.kappa = kappa
        self.base_order_size = base_order_size
        self.max_inventory = max_inventory

    def compute_reservation_price(
        self,
        fair_value: float,
        inventory: float,
        sigma: float,
        time_to_expiry: float,
    ) -> float:
        """Compute the reservation price.

        r = p_hat - q * gamma * sigma^2 * (T - t)

        When long (q > 0): r decreases -> less eager to buy, more to sell.
        When short (q < 0): r increases -> more eager to buy, less to sell.
        """
        r = fair_value - inventory * self.gamma * (sigma ** 2) * time_to_expiry
        return self._clamp(r)

    def compute_optimal_spread(
        self,
        sigma: float,
        time_to_expiry: float,
    ) -> float:
        """Compute the optimal bid-ask spread.

        delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)

        Components:
        - First term: inventory risk compensation (increases with volatility)
        - Second term: spread for capturing order flow profit
        """
        inventory_risk = self.gamma * (sigma ** 2) * time_to_expiry
        order_flow = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return inventory_risk + order_flow

    def generate_quotes(
        self,
        fair_values: dict[str, float],
        inventories: dict[str, float],
        sigmas: dict[str, float],
        time_to_expiry: float,
        token_ids: dict[str, str],
    ) -> list[ASQuote]:
        """Generate bid/ask quotes for all outcomes.

        Args:
            fair_values: {outcome: probability}
            inventories: {outcome: current_inventory}
            sigmas: {outcome: volatility}
            time_to_expiry: Days to expiry.
            token_ids: {outcome: token_id}

        Returns:
            List of ASQuote objects.
        """
        quotes = []

        for outcome in fair_values:
            fv = fair_values[outcome]
            inv = inventories.get(outcome, 0.0)
            sigma = sigmas.get(outcome, 0.05)
            token_id = token_ids.get(outcome, "")

            # Core A-S calculations
            r = self.compute_reservation_price(fv, inv, sigma, time_to_expiry)
            delta = self.compute_optimal_spread(sigma, time_to_expiry)

            bid = self._clamp(r - delta / 2.0)
            ask = self._clamp(r + delta / 2.0)

            # Ensure bid < ask
            if bid >= ask:
                mid = (bid + ask) / 2.0
                bid = self._clamp(mid - 0.005)
                ask = self._clamp(mid + 0.005)

            # Adjust sizes based on inventory
            bid_size = self._adjust_size(inv, "bid")
            ask_size = self._adjust_size(inv, "ask")

            quotes.append(ASQuote(
                outcome=outcome,
                token_id=token_id,
                bid_price=bid,
                ask_price=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                reservation_price=r,
                spread=ask - bid,
                fair_value=fv,
                inventory=inv,
            ))

            logger.debug(
                f"  {outcome}: fv={fv:.4f} r={r:.4f} bid={bid:.4f} ask={ask:.4f} "
                f"spread={ask-bid:.4f} inv={inv:.1f}"
            )

        return quotes

    def _adjust_size(self, inventory: float, side: str) -> float:
        """Adjust order size based on inventory.

        Less eager to buy more when already long (and vice versa).
        """
        if self.max_inventory == 0:
            return self.base_order_size

        ratio = inventory / self.max_inventory

        if side == "bid":
            # Reduce bid size when long
            factor = max(0.1, 1.0 - ratio)
        else:
            # Increase ask size when long (more eager to sell)
            factor = max(0.1, 1.0 + ratio)

        return self.base_order_size * factor

    @staticmethod
    def _clamp(price: float) -> float:
        """Clamp price to valid prediction market range."""
        return max(PRICE_FLOOR, min(PRICE_CEILING, price))
