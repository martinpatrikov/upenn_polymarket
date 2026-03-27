"""HTTP client for the Rust polyfill-rs sidecar service."""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from config.settings import OrderBookSnapshot

logger = logging.getLogger(__name__)


class SidecarClient:
    """Talks to the Rust sidecar for real-time orderbook data."""

    def __init__(self, host: str = "http://localhost:8080"):
        self.host = host.rstrip("/")
        self._http = httpx.Client(timeout=5.0)

    def health_check(self) -> bool:
        try:
            resp = self._http.get(f"{self.host}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def subscribe(self, token_ids: list[str]) -> dict:
        try:
            resp = self._http.post(
                f"{self.host}/subscribe",
                json={"token_ids": token_ids},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return {}

    def unsubscribe(self, token_ids: list[str]) -> dict:
        try:
            resp = self._http.post(
                f"{self.host}/unsubscribe",
                json={"token_ids": token_ids},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return {}

    def get_orderbook(self, token_id: str) -> Optional[OrderBookSnapshot]:
        try:
            resp = self._http.get(f"{self.host}/book/{token_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            return OrderBookSnapshot(
                token_id=token_id,
                bids=[(b["price"], b["size"]) for b in data.get("bids", [])],
                asks=[(a["price"], a["size"]) for a in data.get("asks", [])],
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.error(f"Get orderbook failed for {token_id}: {e}")
            return None

    def get_summary(self, token_id: str) -> Optional[dict]:
        try:
            resp = self._http.get(f"{self.host}/book/{token_id}/summary")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Get summary failed for {token_id}: {e}")
            return None

    def get_all_books(self) -> list[dict]:
        try:
            resp = self._http.get(f"{self.host}/books")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Get all books failed: {e}")
            return []

    def get_midpoint(self, token_id: str) -> Optional[float]:
        summary = self.get_summary(token_id)
        if summary:
            return summary.get("midpoint")
        return None

    def get_spread(self, token_id: str) -> Optional[float]:
        summary = self.get_summary(token_id)
        if summary:
            return summary.get("spread")
        return None
