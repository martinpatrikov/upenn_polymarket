"""Portfolio-level risk management and trade validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config.settings import PortfolioState, TradeDecision, Side

logger = logging.getLogger(__name__)


@dataclass
class RiskCheck:
    approved: bool
    reason: str


@dataclass
class StopLossAction:
    token_id: str
    market_question: str
    current_loss_pct: float
    action: str  # "close"


class RiskManager:
    """Enforces portfolio-level risk constraints on proposed trades."""

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_portfolio_exposure: float = 0.30,
        stop_loss_pct: float = 0.15,
    ):
        self.max_position_pct = max_position_pct
        self.max_portfolio_exposure = max_portfolio_exposure
        self.stop_loss_pct = stop_loss_pct

    def check_trade(
        self, trade: TradeDecision, portfolio: PortfolioState
    ) -> RiskCheck:
        """Validate that a proposed trade does not violate risk limits.

        Checks:
        1. Position size does not exceed max_position_pct of portfolio.
        2. Total portfolio exposure (sum of positions / portfolio value) stays below limit.
        3. Portfolio has sufficient cash for the trade.
        """
        portfolio_value = portfolio.total_value
        if portfolio_value <= 0:
            return RiskCheck(False, "Portfolio value is zero or negative")

        # Check 1: Position size limit
        position_pct = trade.size / portfolio_value if portfolio_value > 0 else 1.0
        if position_pct > self.max_position_pct:
            return RiskCheck(
                False,
                f"Position size {position_pct:.1%} exceeds max {self.max_position_pct:.1%}",
            )

        # Check 2: Total portfolio exposure
        current_exposure = sum(
            pos.get("qty", 0) * pos.get("avg_price", 0)
            for pos in portfolio.positions.values()
        )
        new_exposure = current_exposure + trade.size
        exposure_pct = new_exposure / portfolio_value if portfolio_value > 0 else 1.0
        if exposure_pct > self.max_portfolio_exposure:
            return RiskCheck(
                False,
                f"Portfolio exposure {exposure_pct:.1%} would exceed max {self.max_portfolio_exposure:.1%}",
            )

        # Check 3: Sufficient cash
        if trade.size > portfolio.cash:
            return RiskCheck(
                False,
                f"Insufficient cash: need ${trade.size:.2f}, have ${portfolio.cash:.2f}",
            )

        return RiskCheck(True, "Trade approved")

    def check_stop_losses(
        self, portfolio: PortfolioState, current_prices: dict[str, float]
    ) -> list[StopLossAction]:
        """Check if any positions should be closed due to stop-loss trigger.

        Args:
            portfolio: Current portfolio state.
            current_prices: Dict of token_id -> current price.

        Returns:
            List of positions that should be closed.
        """
        actions: list[StopLossAction] = []

        for token_id, pos in portfolio.positions.items():
            current_price = current_prices.get(token_id, pos.get("avg_price", 0))
            avg_price = pos.get("avg_price", 0)
            qty = pos.get("qty", 0)

            if avg_price <= 0 or qty <= 0:
                continue

            # Compute unrealized loss percentage
            loss_pct = (avg_price - current_price) / avg_price

            if loss_pct >= self.stop_loss_pct:
                actions.append(
                    StopLossAction(
                        token_id=token_id,
                        market_question=pos.get("market_question", "Unknown"),
                        current_loss_pct=loss_pct,
                        action="close",
                    )
                )
                logger.warning(
                    f"STOP LOSS triggered for {token_id}: "
                    f"loss={loss_pct:.1%} >= threshold={self.stop_loss_pct:.1%}"
                )

        return actions
