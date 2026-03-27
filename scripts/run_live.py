#!/usr/bin/env python3
"""Run the trading agent in live simulation mode.

Connects to real Polymarket orderbooks and RSS news feeds.
Paper trades only — no real orders are placed.

Usage:
    python scripts/run_live.py
    python scripts/run_live.py --interval 60
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from config.settings import Settings
from agent.core import TradingAgent

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Agent - Live Simulation")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between analysis cycles (default: 300)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = Settings()
    agent = TradingAgent(config)
    agent.start_live(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
