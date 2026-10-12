import math

import numpy as np
import pandas as pd
import pytest

from bot.indicators import (
    add_vol_features,
    atr,
    ema,
    realized_vol,
    rolling_high,
    rolling_low,
    rsi,
    sma,
    rolling_std,
)


def test_sma_known_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = sma(s, 2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == [1.5, 2.5, 3.5]


def test_rolling_std_sample():
    s = pd.Series([1.0, 2.0, 3.0])
    out = rolling_std(s, 3)
    assert out.iloc[2] == pytest.approx(1.0)


def test_ema_recursive_definition():
    # span=2 -> alpha=2/3; e3 = 23/9 by hand
    s = pd.Series([1.0, 2.0, 3.0])
    out = ema(s, 2)
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(5 / 3)
    assert out.iloc[2] == pytest.approx(23 / 9)


def test_ema_constant_series_is_constant():
    s = pd.Series([5.0] * 50)
    assert ema(s, 10).iloc[-1] == pytest.approx(5.0)


def test_atr_constant_range():
    # Every bar: high = close + 1, low = close - 1, close flat -> TR = 2 -> ATR = 2
    n = 30
    df = pd.DataFrame(
        {"high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n}
    )
    out = atr(df, 14)
    assert out.iloc[-1] == pytest.approx(2.0)


def test_atr_uses_gap_from_prev_close():
    # Gap up: TR of second bar is |high - prev_close| = 10, not high-low = 2
    df = pd.DataFrame(
        {"high": [101.0, 110.0], "low": [99.0, 108.0], "close": [100.0, 109.0]}
    )
    out = atr(df, 1)  # alpha=1 -> ATR equals current TR
    assert out.iloc[1] == pytest.approx(10.0)


def test_rolling_high_low():
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    assert rolling_high(s, 3).iloc[-1] == 5.0
    assert rolling_low(s, 3).iloc[-1] == 2.0


def test_rsi_all_gains_is_100():
    s = pd.Series([float(i) for i in range(1, 30)])
    assert rsi(s, 2).iloc[-1] == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    s = pd.Series([float(i) for i in range(30, 1, -1)])
    assert rsi(s, 2).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_rsi_alternation_converges_to_two_thirds_after_up_move():
    # +1/-1 alternation with alpha=1/2: avg gain 2/3, avg loss 1/3 -> RSI 66.67
    s = pd.Series([100.0, 101.0] * 30)
    assert rsi(s, 2).iloc[-1] == pytest.approx(200 / 3, abs=0.5)


def test_realized_vol_hand_computed():
    # returns [+0.10, -0.10]; sample std = sqrt(0.02); annualized x sqrt(252)
    s = pd.Series([100.0, 110.0, 99.0])
    out = realized_vol(s, 2)
    assert out.iloc[2] == pytest.approx(math.sqrt(0.02) * math.sqrt(252))


def test_realized_vol_nan_head_equals_period():
    # pct_change drops one obs, then rolling(period) needs `period` more:
    # first valid sample lands at index == period.
    s = pd.Series([float(i) for i in range(1, 11)])
    out = realized_vol(s, 3)
    assert out.iloc[:3].isna().all()
    assert not np.isnan(out.iloc[3])


def test_realized_vol_annualization_factor():
    s = pd.Series([100.0, 102.0, 101.0, 104.0, 103.5, 105.0, 104.0])
    expected = s.pct_change().rolling(4).std() * math.sqrt(252)
    pd.testing.assert_series_equal(realized_vol(s, 4), expected)


def test_realized_vol_constant_series_is_zero():
    s = pd.Series([100.0] * 30)
    assert realized_vol(s, 20).iloc[-1] == pytest.approx(0.0)


def _vol_bars(closes):
    return pd.DataFrame({"close": list(closes)})


def test_add_vol_features_adds_columns():
    df = add_vol_features(_vol_bars(np.linspace(100, 120, 60)), lookback=5, rank_window=20)
    assert "realized_vol" in df.columns
    assert "vol_pct" in df.columns


def test_add_vol_features_percentile_range_in_zero_to_one():
    rng = np.random.RandomState(0)
    closes = 100 + np.cumsum(rng.randn(200))
    df = add_vol_features(_vol_bars(closes), lookback=10, rank_window=60)
    vp = df["vol_pct"].dropna()
    assert (vp > 0.0).all() and (vp <= 1.0).all()


def test_add_vol_features_flags_high_vol_regime():
    # Calm -> violent -> calm. vol_pct ranks the current vol within its trailing
    # window, so it peaks at the storm onset (a high reading against a still
    # calm-dominated window) and bottoms out as calm returns after the storm.
    rng = np.random.RandomState(1)
    calm1 = 100 + np.cumsum(rng.randn(150) * 0.1)
    storm = calm1[-1] + np.cumsum(rng.randn(80) * 5.0)
    calm2 = storm[-1] + np.cumsum(rng.randn(80) * 0.1)
    df = add_vol_features(_vol_bars(np.concatenate([calm1, storm, calm2])),
                          lookback=10, rank_window=60)
    assert df["vol_pct"].iloc[155:185].max() >= 0.95   # storm ranks near the top
    assert df["vol_pct"].iloc[235:260].min() < 0.2      # calm-after-storm near the bottom


def test_add_vol_features_nan_head_length():
    # realized_vol is NaN for the first `lookback` bars; the rolling rank then
    # needs a full `rank_window` of non-NaN values, so the first valid vol_pct
    # lands at index lookback + rank_window - 1.
    lookback, rank_window = 5, 20
    df = add_vol_features(_vol_bars(np.linspace(100, 200, 60)), lookback, rank_window)
    first_valid = lookback + rank_window - 1
    assert df["vol_pct"].iloc[:first_valid].isna().all()
    assert not np.isnan(df["vol_pct"].iloc[first_valid])
