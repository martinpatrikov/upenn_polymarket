"""Orderbook microstructure analysis for execution quality."""

from __future__ import annotations

from config.settings import OrderBookSnapshot, OrderBookAnalysis


class OrderBookAnalyzer:
    """Analyzes orderbook depth, spread, and slippage for trade execution."""

    SLIPPAGE_TEST_SIZES = [50.0, 100.0, 250.0, 500.0, 1000.0]

    def analyze(self, book: OrderBookSnapshot) -> OrderBookAnalysis:
        """Full orderbook analysis including spread, depth, and slippage estimates."""
        bid_depth = sum(size for _, size in book.bids)
        ask_depth = sum(size for _, size in book.asks)
        total_depth = bid_depth + ask_depth

        imbalance = bid_depth / ask_depth if ask_depth > 0 else float("inf")

        # Estimate slippage at standard test sizes
        slippage_estimates = {}
        for size in self.SLIPPAGE_TEST_SIZES:
            buy_slippage = self.estimate_slippage(book, "BUY", size)
            sell_slippage = self.estimate_slippage(book, "SELL", size)
            slippage_estimates[size] = max(buy_slippage, sell_slippage)

        return OrderBookAnalysis(
            spread=book.spread,
            midpoint=book.midpoint,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            bid_depth_total=bid_depth,
            ask_depth_total=ask_depth,
            imbalance_ratio=imbalance,
            slippage_estimates=slippage_estimates,
        )

    def estimate_slippage(
        self, book: OrderBookSnapshot, side: str, size_dollars: float
    ) -> float:
        """Estimate slippage by walking through orderbook levels.

        Walks through the relevant side of the book (asks for BUY, bids for SELL)
        accumulating fills until the target size is reached. Returns the
        volume-weighted average fill price minus the midpoint.

        Args:
            book: Current orderbook snapshot.
            side: "BUY" or "SELL".
            size_dollars: Dollar size of the order.

        Returns:
            Estimated slippage in price units (cents). Positive means worse execution.
        """
        levels = book.asks if side == "BUY" else book.bids
        if not levels:
            return 0.0

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

        if total_qty == 0:
            return 0.0

        vwap = total_cost / total_qty
        slippage = abs(vwap - book.midpoint)
        return slippage

    def is_liquid_enough(
        self, analysis: OrderBookAnalysis, min_depth: float = 50.0, max_spread: float = 0.10
    ) -> bool:
        """Check if market has sufficient liquidity for trading.

        Args:
            analysis: Orderbook analysis result.
            min_depth: Minimum total depth on each side in dollars.
            max_spread: Maximum acceptable bid-ask spread.
        """
        return (
            analysis.bid_depth_total >= min_depth
            and analysis.ask_depth_total >= min_depth
            and analysis.spread <= max_spread
        )
