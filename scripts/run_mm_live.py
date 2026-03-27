#!/usr/bin/env python3
"""Run the Avellaneda-Stoikov market making agent in live simulation mode.

Usage:
    python scripts/run_mm_live.py [--interval 30]

Requires the Rust sidecar to be running (optional but recommended):
    ./scripts/start_sidecar.sh
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from config.settings import Settings
from agent.core import MarketMakingAgent


def main():
    parser = argparse.ArgumentParser(description="A-S Market Making Agent")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Polling interval in seconds (default: from config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    args = parser.parse_args()

    config = Settings()
    if args.log_level:
        config.log_level = args.log_level

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    agent = MarketMakingAgent(config)
    agent.start_live(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
