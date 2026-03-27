"""Avellaneda-Stoikov market making model adapted for prediction markets.

Adapted for [0,1]-bounded prediction markets with multi-outcome structure.
Uses normalized time and calibrated parameters for 2-10 cent spreads.

Core formulas:
    Reservation price: r = p_hat - q * gamma * sigma^2 * T_norm
    Optimal spread: delta = gamma * sigma^2 * T_norm + (2/gamma) * ln(1 + gamma/kappa)
    Bid: r - delta/2
    Ask: r + delta/2

Where T_norm = (T-t) / T_max normalizes time to [0,1].
"""

import math
import logging

from config.settings import ASQuote

logger = logging.getLogger(__name__)

PRICE_FLOOR = 0.01
PRICE_CEILING = 0.99

# Max horizon for time normalization (days).
# Beyond this, time scaling saturates to avoid extreme spread widening.
T_MAX_DAYS = 120.0

# Minimum spread (1 cent) to always capture at least some edge.
MIN_SPREAD = 0.01


class AvellanedaStoikov:
    """Avellaneda-Stoikov market making model for prediction markets.

    Default parameters are calibrated for Polymarket prediction markets:
    - gamma=0.5: moderate risk aversion (balances inventory control vs spread)
    - kappa=40: high order arrival rate (prediction markets have decent flow)
    - These produce ~4-8 cent spreads depending on volatility and time.
    """

    def __init__(
        self,
        gamma: float = 0.5,
        kappa: float = 40.0,
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

        r = p_hat - q * gamma * sigma^2 * T_norm

        Uses normalized time T_norm = min(T-t, T_MAX) / T_MAX to keep
        the inventory adjustment bounded for long-dated markets.
        """
        t_norm = self._normalize_time(time_to_expiry)
        r = fair_value - inventory * self.gamma * (sigma ** 2) * t_norm
        return self._clamp(r)

    def compute_optimal_spread(
        self,
        sigma: float,
        time_to_expiry: float,
    ) -> float:
        """Compute the optimal bid-ask spread.

        delta = gamma * sigma^2 * T_norm + (2/gamma) * ln(1 + gamma/kappa)

        With gamma=0.5, kappa=40:
        - Order flow term: (2/0.5)*ln(1+0.5/40) = 4*0.0124 = 0.050
        - Inventory risk: 0.5 * sigma^2 * T_norm (small contribution)
        - Total: ~5 cents base + volatility adjustment
        """
        t_norm = self._normalize_time(time_to_expiry)
        inventory_risk = self.gamma * (sigma ** 2) * t_norm
        order_flow = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        return max(MIN_SPREAD, inventory_risk + order_flow)

    def generate_quotes(
        self,
        fair_values: dict[str, float],
        inventories: dict[str, float],
        sigmas: dict[str, float],
        time_to_expiry: float,
        token_ids: dict[str, str],
    ) -> list[ASQuote]:
        """Generate bid/ask quotes for all outcomes."""
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

            # Ensure bid < ask with minimum 1-cent gap
            if bid >= ask:
                mid = (bid + ask) / 2.0
                bid = self._clamp(mid - MIN_SPREAD / 2)
                ask = self._clamp(mid + MIN_SPREAD / 2)

            # Adjust sizes based on inventory
            bid_size = self._adjust_size(inv, "bid")
            ask_size = self._adjust_size(inv, "ask")

            quotes.append(ASQuote(
                outcome=outcome,
                token_id=token_id,
                bid_price=round(bid, 4),
                ask_price=round(ask, 4),
                bid_size=round(bid_size, 2),
                ask_size=round(ask_size, 2),
                reservation_price=round(r, 4),
                spread=round(ask - bid, 4),
                fair_value=round(fv, 4),
                inventory=round(inv, 2),
            ))

        return quotes

    def _adjust_size(self, inventory: float, side: str) -> float:
        """Adjust order size based on inventory.

        Less eager to buy more when already long (and vice versa).
        """
        if self.max_inventory == 0:
            return self.base_order_size

        ratio = inventory / self.max_inventory

        if side == "bid":
            factor = max(0.1, 1.0 - ratio)
        else:
            factor = max(0.1, 1.0 + ratio)

        return self.base_order_size * factor

    @staticmethod
    def _normalize_time(days_to_expiry: float) -> float:
        """Normalize time to [0, 1] range.

        This prevents the inventory risk and spread terms from exploding
        for long-dated prediction markets (30-120 day horizons).
        """
        return min(days_to_expiry, T_MAX_DAYS) / T_MAX_DAYS

    @staticmethod
    def _clamp(price: float) -> float:
        """Clamp price to valid prediction market range."""
        return max(PRICE_FLOOR, min(PRICE_CEILING, price))
