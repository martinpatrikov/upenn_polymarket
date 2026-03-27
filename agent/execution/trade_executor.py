"""Trade executor with order type selection and paper trading."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import (
    ExecutionResult,
    OrderBookSnapshot,
    OrderType,
    Side,
    TradeDecision,
)

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Decides order type and executes (or simulates) trades.

    Order Type Selection Logic:
    - Strong edge (>10%) + liquid market: Market order (FOK) for immediate fill
    - Moderate edge (3-10%): Limit order (GTC) placed between midpoint and best bid/ask
      to capture more edge while accepting the risk of non-fill.

    In simulation mode (default), no actual orders are placed. Instead, fill prices
    are estimated by walking through the orderbook.
    """

    # Edge thresholds for order type selection
    MARKET_ORDER_EDGE_THRESHOLD = 0.10

    def __init__(self, mode: str = "simulation"):
        self.mode = mode
        self.trade_log: list[ExecutionResult] = []

    def execute(
        self,
        decision: TradeDecision,
        book: OrderBookSnapshot,
    ) -> ExecutionResult:
        """Execute (or simulate) a trade based on the decision and orderbook state.

        Args:
            decision: The trade decision from the scoring engine.
            book: Current orderbook snapshot for fill estimation.

        Returns:
            ExecutionResult with fill details.
        """
        edge = abs(decision.effective_edge)

        # Select order type based on edge magnitude
        if edge > self.MARKET_ORDER_EDGE_THRESHOLD and self._has_liquidity(book, decision.side):
            order_type = OrderType.MARKET
            fill_price = self._simulate_market_fill(book, decision.side, decision.size)
            reasoning = [
                f"Edge {edge:.1%} > {self.MARKET_ORDER_EDGE_THRESHOLD:.0%}: using MARKET order",
                f"Estimated fill at {fill_price:.4f}",
            ]
        else:
            order_type = OrderType.LIMIT
            fill_price = self._compute_limit_price(book, decision.side)
            reasoning = [
                f"Edge {edge:.1%}: using LIMIT order",
                f"Limit price set at {fill_price:.4f}",
            ]

        slippage = abs(fill_price - book.midpoint) if fill_price else 0.0

        result = ExecutionResult(
            order_type=order_type,
            price=fill_price,
            size=decision.size,
            side=decision.side,
            simulated=(self.mode == "simulation"),
            timestamp=datetime.now(timezone.utc),
            reasoning=decision.reasoning + reasoning,
            slippage=slippage,
        )

        self.trade_log.append(result)

        if self.mode == "simulation":
            logger.info(
                f"[PAPER TRADE] {decision.side.value} {order_type.value} | "
                f"${decision.size:.2f} @ {fill_price:.4f} | "
                f"slippage={slippage:.4f}"
            )
        else:
            logger.info(
                f"[LIVE TRADE] {decision.side.value} {order_type.value} | "
                f"${decision.size:.2f} @ {fill_price:.4f}"
            )

        return result

    def _simulate_market_fill(
        self, book: OrderBookSnapshot, side: Side, size_dollars: float
    ) -> float:
        """Estimate fill price for a market order by walking the book."""
        levels = book.asks if side == Side.BUY else book.bids
        if not levels:
            return book.midpoint

        remaining = size_dollars
        total_cost = 0.0
        total_qty = 0.0

        for price, qty in levels:
            if remaining <= 0:
                break
            level_value = price * qty
            fill_value = min(remaining, level_value)
            fill_qty = fill_value / price if price > 0 else 0
            total_cost += fill_value
            total_qty += fill_qty
            remaining -= fill_value

        return total_cost / total_qty if total_qty > 0 else book.midpoint

    def _compute_limit_price(self, book: OrderBookSnapshot, side: Side) -> float:
        """Compute a limit price between midpoint and best bid/ask.

        For BUY: place limit between midpoint and best_ask (captures some spread).
        For SELL: place limit between best_bid and midpoint.
        """
        mid = book.midpoint
        if side == Side.BUY:
            # Place slightly below best ask to capture spread
            best_ask = book.best_ask
            return round(mid + (best_ask - mid) * 0.4, 4)
        else:
            # Place slightly above best bid
            best_bid = book.best_bid
            return round(best_bid + (mid - best_bid) * 0.6, 4)

    def _has_liquidity(
        self, book: OrderBookSnapshot, side: Side, min_depth: float = 50.0
    ) -> bool:
        """Check if there's enough liquidity for a market order."""
        levels = book.asks if side == Side.BUY else book.bids
        total = sum(price * qty for price, qty in levels)
        return total >= min_depth
