"""Kelly Criterion position sizing for binary prediction markets.

Mathematical Framework
======================

For a binary prediction market where you buy a contract at price p and
your estimated true probability is q:

LONG (buying YES at price p):
  - If correct: receive $1, profit = (1 - p), return = (1 - p) / p
  - If incorrect: lose $p
  - Win payoff b = (1 - p) / p
  - Kelly fraction: f* = (b*q - (1-q)) / b
  - Simplifies to: f* = (q - p) / (1 - p)

SHORT (buying NO, equivalent to buying complement at 1-p):
  - Equivalent to buying YES of complement at price (1-p) with probability (1-q).
  - f* = ((1-q) - (1-p)) / (1 - (1-p)) = (p - q) / p

FRACTIONAL KELLY:
  Full Kelly maximizes long-run growth but with high variance.
  We use quarter-Kelly (fraction=0.25) which retains ~56% of expected growth
  with dramatically reduced risk of ruin.

  f_actual = kelly_fraction * f*

POSITION CAP:
  Additional safety via max_position_pct (default 5% of portfolio).
"""

from __future__ import annotations

from config.settings import PositionSize, Side


class KellyCriterion:
    """Position sizing using the Kelly Criterion adapted for prediction markets."""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.05,
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct

    def compute_position_size(
        self,
        portfolio_value: float,
        estimated_prob: float,
        market_price: float,
        side: Side,
    ) -> PositionSize:
        """Compute the optimal position size.

        Args:
            portfolio_value: Total portfolio value in dollars.
            estimated_prob: Agent's estimated probability of YES outcome.
            market_price: Current market price (probability) of YES token.
            side: BUY (long YES) or SELL (long NO).

        Returns:
            PositionSize with raw Kelly, scaled Kelly, dollar amount, and contracts.
        """
        kelly_raw = self._raw_kelly(estimated_prob, market_price, side)

        # If Kelly is negative or zero, no trade
        if kelly_raw <= 0:
            return PositionSize(
                kelly_raw=kelly_raw,
                kelly_scaled=0.0,
                dollar_amount=0.0,
                num_contracts=0.0,
                side=side,
            )

        # Apply fractional Kelly
        kelly_scaled = self.kelly_fraction * kelly_raw

        # Cap at max position
        kelly_scaled = min(kelly_scaled, self.max_position_pct)

        # Dollar amount
        dollar_amount = portfolio_value * kelly_scaled

        # Number of contracts (at current market price)
        price = market_price if side == Side.BUY else (1 - market_price)
        num_contracts = dollar_amount / price if price > 0 else 0.0

        return PositionSize(
            kelly_raw=kelly_raw,
            kelly_scaled=kelly_scaled,
            dollar_amount=dollar_amount,
            num_contracts=num_contracts,
            side=side,
        )

    def _raw_kelly(
        self, estimated_prob: float, market_price: float, side: Side
    ) -> float:
        """Compute raw (full) Kelly fraction.

        For BUY (long YES): f* = (q - p) / (1 - p)
        For SELL (long NO): f* = (p - q) / p
        """
        q = estimated_prob
        p = market_price

        # Clamp to avoid division issues
        p = max(0.01, min(0.99, p))
        q = max(0.01, min(0.99, q))

        if side == Side.BUY:
            # Long YES: profit if event occurs
            return (q - p) / (1 - p)
        else:
            # Long NO: profit if event does not occur
            return (p - q) / p
