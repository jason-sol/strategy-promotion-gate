from collections.abc import Callable

from config import RISK, InstrumentConfig, StrategyName

from bot.strategies.base import Strategy
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.momentum_breakout import MomentumBreakout
from bot.strategies.pead_drift import PeadDrift
from bot.strategies.rsi_mean_reversion import RsiMeanReversion
from bot.strategies.trend_following import TrendFollowing
from bot.strategies.ts_momentum import TsMomentum
from bot.strategies.turn_of_month import TurnOfMonth


def _vol_gate_kwargs(p: dict) -> dict:
    # vol_gate_percentile is a RELATIVE rank (0-1) of current realized vol
    # within the trailing vol_gate_rank_window, not an absolute vol level:
    # the gate self-recalibrates as the regime shifts.
    return {
        "vol_gate_percentile": p.get("vol_gate_percentile"),
        "vol_gate_lookback": p.get("vol_gate_lookback", 20),
        "vol_gate_rank_window": p.get("vol_gate_rank_window", 252),
    }


def _mean_reversion(p: dict) -> Strategy:
    return MeanReversion(
        sma_period=p["sma_period"], entry_z=p["entry_z"],
        allow_short=p["allow_short"], atr_period=RISK.atr_period,
    )


def _momentum_breakout(p: dict) -> Strategy:
    return MomentumBreakout(
        channel_period=p["channel_period"], volume_mult=p["volume_mult"],
        trail_atr_mult=p["trail_atr_mult"], atr_period=RISK.atr_period,
    )


def _rsi_mean_reversion(p: dict) -> Strategy:
    return RsiMeanReversion(
        rsi_period=p["rsi_period"], entry_below=p["entry_below"],
        regime_period=p["regime_period"], exit_period=p["exit_period"],
        stop_mult=p["stop_mult"], atr_period=RISK.atr_period,
        confirm_days=p.get("confirm_days", 1),
        **_vol_gate_kwargs(p),
    )


def _trend_following(p: dict) -> Strategy:
    return TrendFollowing(
        fast_period=p["fast_period"], slow_period=p["slow_period"],
        trail_atr_mult=p["trail_atr_mult"], atr_period=RISK.atr_period,
        allow_short=p["allow_short"],
        **_vol_gate_kwargs(p),
    )


def _ts_momentum(p: dict) -> Strategy:
    return TsMomentum(
        lookbacks=p.get("lookbacks"),
        skip_days=p.get("skip_days", 21),
        atr_period=RISK.atr_period,
        stop_atr_mult=p.get("stop_atr_mult", 3.0),
        **_vol_gate_kwargs(p),
    )


def _turn_of_month(p: dict) -> Strategy:
    return TurnOfMonth(
        entry_window_bdays=p.get("entry_window_bdays", 2),
        exit_day_of_month=p.get("exit_day_of_month", 3),
        atr_period=RISK.atr_period,
        stop_atr_mult=p.get("stop_atr_mult", 3.0),
        **_vol_gate_kwargs(p),
    )


def _pead_drift(p: dict) -> Strategy:
    # Signal-logging-only sleeve: constructible but not wired into the live
    # lineup. events must be supplied via params (from bot.earnings).
    if "events" not in p:
        raise ValueError("PEAD_DRIFT requires params['events'] (earnings DataFrame)")
    return PeadDrift(
        events=p["events"],
        sue_min=p.get("sue_min", 1.0),
        drift_window=p.get("drift_window", 10),
        atr_period=RISK.atr_period,
        stop_atr_mult=p.get("stop_atr_mult", 3.0),
    )


STRATEGY_BUILDERS: dict[StrategyName, Callable[[dict], Strategy]] = {
    StrategyName.MEAN_REVERSION: _mean_reversion,
    StrategyName.MOMENTUM_BREAKOUT: _momentum_breakout,
    StrategyName.RSI_MEAN_REVERSION: _rsi_mean_reversion,
    StrategyName.TREND_FOLLOWING: _trend_following,
    StrategyName.TS_MOMENTUM: _ts_momentum,
    StrategyName.TURN_OF_MONTH: _turn_of_month,
    StrategyName.PEAD_DRIFT: _pead_drift,
}


def build_strategy(cfg: InstrumentConfig) -> Strategy:
    builder = STRATEGY_BUILDERS.get(cfg.strategy)
    if builder is None:
        raise ValueError(f"unknown strategy {cfg.strategy}")
    return builder(cfg.params)
