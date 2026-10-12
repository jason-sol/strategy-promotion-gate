"""ATR-based sizing, correlation filter, and portfolio drawdown circuit breaker.

Pure logic — shared verbatim by the live engine and the backtester.
"""

import math
from dataclasses import dataclass

import pandas as pd

from bot.indicators import realized_vol_at
from bot.strategies.base import PositionState
from config import AssetClass, RiskConfig


@dataclass(frozen=True)
class SizingResult:
    qty: float
    stop_price: float


@dataclass(frozen=True)
class RiskBudgetCheck:
    realized_pct: float     # fraction of equity lost if the stop fills at its level
    budget_pct: float       # fraction the sizing intended to risk
    within_budget: bool
    note: str


# Whole shares and the notional cap only ever round risk down, so anything above
# the budget is a sizing error rather than arithmetic: the tolerance only has to
# absorb float noise.
RISK_BUDGET_TOLERANCE = 1e-9


def realized_risk_pct(qty: float, price: float, stop_price: float,
                      equity: float) -> float:
    """Fraction of equity a position loses if its stop fills at its level.
    Zero equity has no meaningful fraction, so it reports no risk."""
    if equity <= 0:
        return 0.0
    return abs(qty) * abs(price - stop_price) / equity


def within_risk_budget(realized_pct: float, budget_pct: float) -> bool:
    return realized_pct <= budget_pct + RISK_BUDGET_TOLERANCE


class RiskManager:
    def __init__(self, risk: RiskConfig):
        self.risk = risk

    def realized_vol_for(self, close: pd.Series, i: int, cfg) -> float | None:
        """Single realized-vol entry point for both engines so they cannot drift,
        including the per-instrument opt-in. None when the instrument hasn't opted
        in — sizing then stays neutral."""
        if not cfg.params.get("vol_target"):
            return None
        return realized_vol_at(close, i, self.risk.vol_lookback)

    def vol_target_scalar(self, realized_vol: float | None) -> float:
        """Scale size toward a target portfolio vol: shrink in high-vol regimes,
        grow in calm ones, bounded so it never dominates the per-trade ATR risk.
        NaN/zero/negative/None realized vol is treated as no signal (neutral)."""
        if realized_vol is None or math.isnan(realized_vol) or realized_vol <= 0:
            return 1.0
        raw = self.risk.target_vol_annual / realized_vol
        return max(self.risk.vol_scalar_min, min(raw, self.risk.vol_scalar_max))

    def risk_budget_pct(self, realized_vol: float | None) -> float:
        """Per-trade risk a sizing is allowed to take: the configured fraction
        carrying the same bounded vol-target scalar the sizing applied. The flat
        fraction alone is not the invariant wherever vol targeting is on."""
        return self.risk.risk_per_trade * self.vol_target_scalar(realized_vol)

    def check_risk_budget(self, qty: float, price: float, stop_price: float,
                          equity: float,
                          realized_vol: float | None = None) -> RiskBudgetCheck:
        """Measure what a size actually risks against what it was meant to.
        Reports; never gates — callers log the verdict and submit either way."""
        realized = realized_risk_pct(qty, price, stop_price, equity)
        budget = self.risk_budget_pct(realized_vol)
        if within_risk_budget(realized, budget):
            return RiskBudgetCheck(realized, budget, True, "within budget")
        over_bps = (realized - budget) * 10_000
        return RiskBudgetCheck(realized, budget, False,
                               f"over budget by {over_bps:.1f}bps of equity")

    def position_size(self, equity: float, price: float, atr_value: float,
                      side: str, asset_class: AssetClass,
                      stop_atr_mult: float | None = None,
                      realized_vol: float | None = None) -> SizingResult:
        stop_distance = (stop_atr_mult or self.risk.stop_atr_mult) * atr_value
        if stop_distance <= 0 or price <= 0 or equity <= 0:
            return SizingResult(qty=0, stop_price=0)

        risk_dollars = equity * self.risk.risk_per_trade
        qty = risk_dollars / stop_distance
        # Vol targeting layers a portfolio-vol adjustment on top of per-trade ATR
        # risk; both shrink size in volatile regimes, so this deliberately
        # double-counts. The bounded scalar and per-instrument opt-in contain it.
        qty *= self.vol_target_scalar(realized_vol)
        qty = min(qty, self.risk.max_position_notional_pct * equity / price)

        if asset_class == AssetClass.US_EQUITY:
            # Whole shares: fractional qty is incompatible with broker-side
            # stop orders and fractional shorts are not allowed at all.
            qty = float(math.floor(qty))
        else:
            qty = round(qty, 6)

        stop_price = price - stop_distance if side == "long" else price + stop_distance
        return SizingResult(qty=qty, stop_price=stop_price)

    def approve_entry(self, symbol: str, side: str, equity: float, price: float,
                      atr_value: float, asset_class: AssetClass,
                      positions: dict[str, PositionState],
                      stop_atr_mult: float | None = None,
                      realized_vol: float | None = None) -> SizingResult | None:
        """Full entry gate shared by live and backtest: correlation filter then
        sizing. None means don't trade."""
        if self.correlation_blocked(symbol, side, positions):
            return None
        sizing = self.position_size(equity, price, atr_value, side, asset_class,
                                    stop_atr_mult, realized_vol)
        return sizing if sizing.qty > 0 else None

    def correlation_blocked(self, symbol: str, side: str,
                            positions: dict[str, PositionState]) -> bool:
        """Block new BTC longs when SPY and QQQ are both long: three correlated
        risk-on positions would double up real exposure."""
        if symbol != "BTC/USD" or side != "long":
            return False
        spy = positions.get("SPY")
        qqq = positions.get("QQQ")
        return spy is not None and spy.side == "long" and qqq is not None and qqq.side == "long"

    def drawdown_halted(self, equity: float, peak_equity: float) -> bool:
        return equity < (1 - self.risk.max_drawdown) * peak_equity
