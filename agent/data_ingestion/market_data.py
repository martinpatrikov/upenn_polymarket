"""Polymarket CLOB and Gamma API client for market and orderbook data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from config.settings import MarketInfo, OrderBookSnapshot

logger = logging.getLogger(__name__)


class MarketDataClient:
    """Wraps Polymarket CLOB API (orderbooks) and Gamma API (market metadata).

    The Gamma API provides market metadata (questions, descriptions, token IDs).
    The CLOB API provides orderbook data (bids, asks, prices).

    Key Gamma API fields:
    - conditionId: unique market identifier
    - clobTokenIds: JSON string array of CLOB token IDs [YES_token, NO_token]
    - outcomes: JSON string array ["Yes", "No"]
    - outcomePrices: JSON string array of current prices ["0.65", "0.35"]
    """

    def __init__(
        self,
        clob_host: str = "https://clob.polymarket.com",
        gamma_host: str = "https://gamma-api.polymarket.com",
    ):
        self.clob_host = clob_host.rstrip("/")
        self.gamma_host = gamma_host.rstrip("/")
        self._http = httpx.Client(timeout=15.0)

        # Try to initialize the CLOB client for orderbook data
        self._clob_client = None
        try:
            from py_clob_client.client import ClobClient
            self._clob_client = ClobClient(clob_host, chain_id=137)
            logger.info("CLOB client initialized (read-only)")
        except Exception as e:
            logger.warning(f"py-clob-client unavailable, using REST fallback: {e}")

    def _parse_market(self, m: dict) -> Optional[MarketInfo]:
        """Parse a market dict from the Gamma API into a MarketInfo object."""
        # Parse outcomes and prices from JSON strings
        try:
            outcomes = json.loads(m.get("outcomes", "[]"))
        except (json.JSONDecodeError, TypeError):
            outcomes = []

        try:
            prices = json.loads(m.get("outcomePrices", "[]"))
        except (json.JSONDecodeError, TypeError):
            prices = []

        try:
            clob_token_ids = json.loads(m.get("clobTokenIds", "[]"))
        except (json.JSONDecodeError, TypeError):
            clob_token_ids = []

        # Build tokens list
        tokens = []
        for i, outcome in enumerate(outcomes):
            token_id = clob_token_ids[i] if i < len(clob_token_ids) else ""
            price = float(prices[i]) if i < len(prices) else 0.0
            if token_id:  # Only include if we have a token ID
                tokens.append({
                    "token_id": token_id,
                    "outcome": outcome,
                    "price": price,
                })

        if not tokens:
            return None

        return MarketInfo(
            condition_id=m.get("conditionId", m.get("condition_id", "")),
            question=m.get("question", ""),
            description=m.get("description", ""),
            tokens=tokens,
            volume=float(m.get("volume", 0) or 0),
            liquidity=float(m.get("liquidity", 0) or 0),
            end_date=m.get("endDate", m.get("end_date_iso")),
            active=m.get("active", True),
        )

    def get_active_markets(self, limit: int = 50, offset: int = 0) -> list[MarketInfo]:
        """Fetch active markets from both Gamma events and CLOB endpoints.

        Strategy:
        1. Gamma /events endpoint for clobTokenIds and metadata
        2. CLOB /markets endpoint for additional active markets with token IDs
        3. Merge, deduplicate, sort by volume
        """
        markets: list[MarketInfo] = []
        seen_conditions: set[str] = set()

        # Source 1: Gamma events endpoint (richer metadata, includes clobTokenIds)
        try:
            resp = self._http.get(
                f"{self.gamma_host}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 50,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            for event in resp.json():
                for m in event.get("markets", []):
                    if not m.get("active", False) or m.get("closed", True):
                        continue
                    market = self._parse_market(m)
                    if market and market.tokens and market.condition_id not in seen_conditions:
                        # Prefer contested markets (price between 0.10 and 0.90)
                        yes_price = market.tokens[0]["price"] if market.tokens else 0
                        if yes_price < 0.05 or yes_price > 0.95:
                            continue
                        seen_conditions.add(market.condition_id)
                        markets.append(market)
        except Exception as e:
            logger.warning(f"Events endpoint failed: {e}")

        # Source 2: CLOB /markets endpoint (has token_id directly)
        try:
            resp = self._http.get(
                f"{self.clob_host}/markets",
                params={"next_cursor": "MA=="},
            )
            resp.raise_for_status()
            data = resp.json()
            for m in data.get("data", []):
                if not m.get("active") or m.get("closed") or m.get("archived"):
                    continue
                if not m.get("accepting_orders"):
                    continue
                cond_id = m.get("condition_id", "")
                if cond_id in seen_conditions:
                    continue

                tokens = []
                for t in m.get("tokens", []):
                    tokens.append({
                        "token_id": t.get("token_id", ""),
                        "outcome": t.get("outcome", ""),
                        "price": float(t.get("price", 0)),
                    })
                if not tokens:
                    continue

                # Filter: only include contested markets (YES price between 0.10 and 0.90)
                yes_price = tokens[0]["price"] if tokens else 0
                if yes_price < 0.10 or yes_price > 0.90:
                    continue

                market = MarketInfo(
                    condition_id=cond_id,
                    question=m.get("question", ""),
                    description=m.get("description", ""),
                    tokens=tokens,
                    volume=0,  # CLOB endpoint doesn't have volume
                    liquidity=0,
                    end_date=m.get("end_date_iso"),
                    active=True,
                )
                seen_conditions.add(cond_id)
                markets.append(market)
        except Exception as e:
            logger.warning(f"CLOB markets endpoint failed: {e}")

        # Sort by: volume first, then prefer prices closer to 0.5 (more interesting/contested)
        def sort_key(m):
            vol = m.volume
            # Price closeness to 0.5 (most uncertain = most interesting)
            yes_price = m.tokens[0]["price"] if m.tokens else 0
            uncertainty = 1.0 - abs(0.5 - yes_price) * 2  # 1.0 at 0.5, 0.0 at 0 or 1
            return (vol, uncertainty)

        markets.sort(key=sort_key, reverse=True)
        logger.info(f"Fetched {len(markets)} total active markets")
        return markets[:limit]

    def get_market_by_id(self, condition_id: str) -> Optional[MarketInfo]:
        """Get a single market by condition ID."""
        try:
            resp = self._http.get(f"{self.gamma_host}/markets/{condition_id}")
            resp.raise_for_status()
            m = resp.json()
            return self._parse_market(m)
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None

    def get_orderbook(self, token_id: str) -> Optional[OrderBookSnapshot]:
        """Get the full orderbook for a token."""
        try:
            if self._clob_client:
                book = self._clob_client.get_order_book(token_id)
                # OrderBookSummary has .bids/.asks as lists of OrderSummary objects
                # OrderSummary has .price and .size as string attributes
                bids = [
                    (float(o.price), float(o.size))
                    for o in (book.bids or [])
                ]
                asks = [
                    (float(o.price), float(o.size))
                    for o in (book.asks or [])
                ]
            else:
                resp = self._http.get(
                    f"{self.clob_host}/book",
                    params={"token_id": token_id},
                )
                resp.raise_for_status()
                data = resp.json()
                bids = [
                    (float(o.get("price", 0)), float(o.get("size", 0)))
                    for o in data.get("bids", [])
                ]
                asks = [
                    (float(o.get("price", 0)), float(o.get("size", 0)))
                    for o in data.get("asks", [])
                ]

            # Sort: bids descending by price, asks ascending
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            return OrderBookSnapshot(
                token_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Failed to get orderbook for {token_id}: {e}")
            return None

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        try:
            if self._clob_client:
                mid = self._clob_client.get_midpoint(token_id)
                return float(mid) if mid else None
            else:
                book = self.get_orderbook(token_id)
                return book.midpoint if book else None
        except Exception as e:
            logger.error(f"Failed to get midpoint for {token_id}: {e}")
            return None

    def get_spread(self, token_id: str) -> Optional[float]:
        """Get the bid-ask spread for a token."""
        book = self.get_orderbook(token_id)
        return book.spread if book else None

    def search_markets(self, query: str, limit: int = 10) -> list[MarketInfo]:
        """Search markets by keyword via Gamma API."""
        try:
            resp = self._http.get(
                f"{self.gamma_host}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": limit,
                    "tag": query,
                },
            )
            resp.raise_for_status()
            markets = []
            for event in resp.json():
                for m in event.get("markets", []):
                    market = self._parse_market(m)
                    if market and market.tokens:
                        markets.append(market)
            return markets
        except Exception as e:
            logger.error(f"Failed to search markets: {e}")
            return []
