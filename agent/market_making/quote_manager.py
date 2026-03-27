"""Quote lifecycle management for the market making agent."""

import logging
import time
from config.settings import ASQuote, Side
from agent.market_making.inventory_manager import InventoryManager

logger = logging.getLogger(__name__)


class QuoteManager:
    """Manages the lifecycle of outstanding quotes and simulates fills.

    Key paper-trading behaviors:
    - Only fills when our quote crosses the market (realistic)
    - Tracks which price levels have already been filled to prevent
      infinite accumulation from stale market prices
    - A quote is "consumed" after it fills — must be regenerated next cycle
    """

    def __init__(
        self,
        inventory_manager: InventoryManager,
        refresh_interval_seconds: float = 30.0,
    ):
        self.inventory = inventory_manager
        self.refresh_interval = refresh_interval_seconds
        self._active_quotes: dict[str, ASQuote] = {}
        self._last_refresh: float = 0.0
        self._fill_history: list[dict] = []
        # Track last fill price per outcome+side to avoid re-filling at same level
        self._last_fill_price: dict[str, float] = {}  # "outcome:BUY" -> price

    def update_quotes(self, new_quotes: list[ASQuote]) -> None:
        self._active_quotes = {q.outcome: q for q in new_quotes}
        self._last_refresh = time.time()

    def needs_refresh(self) -> bool:
        return (time.time() - self._last_refresh) >= self.refresh_interval

    def simulate_fills(
        self,
        orderbooks: dict[str, dict],
    ) -> list[dict]:
        """Simulate fills for paper trading.

        Realistic crossing logic:
        - Our BID gets filled when market best_ask <= our bid (someone sells to us)
        - Our ASK gets filled when market best_bid >= our ask (someone buys from us)
        - A quote at the same price level only fills ONCE (prevents infinite accumulation)
        - After a fill, the quote is consumed; next cycle generates fresh quotes
          with updated reservation prices from the new inventory
        """
        fills = []

        for outcome, quote in self._active_quotes.items():
            book = orderbooks.get(outcome, {})
            mkt_best_ask = book.get("best_ask")
            mkt_best_bid = book.get("best_bid")

            bid_key = f"{outcome}:BUY"
            ask_key = f"{outcome}:SELL"

            # BID fill: market best_ask <= our bid
            if (
                mkt_best_ask is not None
                and mkt_best_ask <= quote.bid_price
                and quote.bid_size > 0
                and not self.inventory.should_reduce(outcome, Side.BUY)
                and self._last_fill_price.get(bid_key) != round(quote.bid_price, 4)
            ):
                fill = {
                    "outcome": outcome,
                    "side": "BUY",
                    "price": quote.bid_price,
                    "size": quote.bid_size,
                    "market_price": mkt_best_ask,
                    "edge": quote.fair_value - quote.bid_price,
                }
                self.inventory.update(outcome, Side.BUY, quote.bid_size, quote.bid_price)
                fills.append(fill)
                self._fill_history.append(fill)
                self._last_fill_price[bid_key] = round(quote.bid_price, 4)

            # ASK fill: market best_bid >= our ask
            if (
                mkt_best_bid is not None
                and mkt_best_bid >= quote.ask_price
                and quote.ask_size > 0
                and not self.inventory.should_reduce(outcome, Side.SELL)
                and self._last_fill_price.get(ask_key) != round(quote.ask_price, 4)
            ):
                fill = {
                    "outcome": outcome,
                    "side": "SELL",
                    "price": quote.ask_price,
                    "size": quote.ask_size,
                    "market_price": mkt_best_bid,
                    "edge": quote.ask_price - quote.fair_value,
                }
                self.inventory.update(outcome, Side.SELL, quote.ask_size, quote.ask_price)
                fills.append(fill)
                self._fill_history.append(fill)
                self._last_fill_price[ask_key] = round(quote.ask_price, 4)

        return fills

    def get_active_quotes(self) -> list[ASQuote]:
        return list(self._active_quotes.values())

    def get_fill_history(self) -> list[dict]:
        return list(self._fill_history)

    def cancel_all(self) -> None:
        self._active_quotes.clear()
        self._last_fill_price.clear()
