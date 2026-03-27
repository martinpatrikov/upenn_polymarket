"""Quote lifecycle management for the market making agent."""

import logging
import time
from config.settings import ASQuote, Side
from agent.market_making.inventory_manager import InventoryManager

logger = logging.getLogger(__name__)


class QuoteManager:
    """Manages the lifecycle of outstanding quotes."""

    def __init__(
        self,
        inventory_manager: InventoryManager,
        refresh_interval_seconds: float = 30.0,
    ):
        self.inventory = inventory_manager
        self.refresh_interval = refresh_interval_seconds
        self._active_quotes: dict[str, ASQuote] = {}  # outcome -> latest quote
        self._last_refresh: float = 0.0
        self._fill_history: list[dict] = []

    def update_quotes(self, new_quotes: list[ASQuote]) -> None:
        """Replace all active quotes with new ones."""
        self._active_quotes = {q.outcome: q for q in new_quotes}
        self._last_refresh = time.time()

    def needs_refresh(self) -> bool:
        """Check if quotes are stale and need refreshing."""
        return (time.time() - self._last_refresh) >= self.refresh_interval

    def simulate_fills(
        self,
        orderbooks: dict[str, dict],
    ) -> list[dict]:
        """Simulate fills for paper trading.

        For each active quote, check if our bid/ask would get hit
        based on the current orderbook state.

        Returns list of simulated fill dicts.
        """
        fills = []

        for outcome, quote in self._active_quotes.items():
            book = orderbooks.get(outcome, {})
            best_ask = book.get("best_ask")
            best_bid = book.get("best_bid")

            # Our bid gets filled if market ask <= our bid
            if best_ask is not None and best_ask <= quote.bid_price and quote.bid_size > 0:
                fill = {
                    "outcome": outcome,
                    "side": "BUY",
                    "price": quote.bid_price,
                    "size": quote.bid_size,
                    "type": "bid_fill",
                }
                self.inventory.update(outcome, Side.BUY, quote.bid_size, quote.bid_price)
                fills.append(fill)
                self._fill_history.append(fill)

            # Our ask gets filled if market bid >= our ask
            if best_bid is not None and best_bid >= quote.ask_price and quote.ask_size > 0:
                fill = {
                    "outcome": outcome,
                    "side": "SELL",
                    "price": quote.ask_price,
                    "size": quote.ask_size,
                    "type": "ask_fill",
                }
                self.inventory.update(outcome, Side.SELL, quote.ask_size, quote.ask_price)
                fills.append(fill)
                self._fill_history.append(fill)

        return fills

    def get_active_quotes(self) -> list[ASQuote]:
        """Get all currently active quotes."""
        return list(self._active_quotes.values())

    def get_fill_history(self) -> list[dict]:
        """Get all historical fills."""
        return list(self._fill_history)

    def cancel_all(self) -> None:
        """Cancel all active quotes."""
        self._active_quotes.clear()
