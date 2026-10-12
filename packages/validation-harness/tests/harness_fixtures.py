"""Synthetic candidates and bars for the harness suite.

The harness only ever sees an equity curve, so the candidates here return curves
directly instead of simulating anything. Every curve is a deterministic function
of the dates it covers, so truncating a frame the way walk_forward does leaves
each date's return unchanged.
"""

import math

import numpy as np
import pandas as pd

from validation_harness import Candidate

INITIAL_EQUITY = 100_000.0


def bars(n: int, freq: str = "B", start: str = "2020-01-01") -> pd.DataFrame:
    """Close-only OHLCV frame. Business days by default: the weekend gaps are
    what resample-to-calendar-day code has to survive, so a gapless index would
    hide the seam."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def compound(steps: pd.Series) -> pd.Series:
    """Equity curve anchored at the starting balance, so the first row prices no
    return and the rest price their own step."""
    anchored = steps.copy()
    anchored.iloc[0] = 0.0
    return INITIAL_EQUITY * (1.0 + anchored).cumprod()


def curve(index: pd.DatetimeIndex, step: float) -> pd.Series:
    return compound(pd.Series(step, index=index))


def drift(label: str, symbol: str = "A", step: float = 0.01,
          skip: int | None = None) -> Candidate:
    """Candidate whose equity compounds at a constant rate from sim_start. `skip`
    drops that positional row from the curve, standing in for a date the
    simulator never priced."""

    def run(frame: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
        index = frame.index[frame.index >= sim_start]
        equity = curve(index, step)
        return equity.drop(index[skip]) if skip is not None else equity

    return Candidate(label, symbol, run)


def noisy(label: str, symbol: str = "A", seed: int = 0, drift_per_bar: float = 0.0,
          scale: float = 0.002) -> Candidate:
    """Candidate with a drifting, dispersed return path. The path is drawn over
    the whole frame and then sliced, so a date keeps its step under truncation."""

    def run(frame: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
        rng = np.random.default_rng(seed)
        steps = pd.Series(
            drift_per_bar + scale * rng.standard_normal(len(frame)), index=frame.index
        )
        return compound(steps[steps.index >= sim_start])

    return Candidate(label, symbol, run)


def empty(label: str, symbol: str = "A") -> Candidate:
    def run(frame: pd.DataFrame, sim_start: pd.Timestamp) -> pd.Series:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

    return Candidate(label, symbol, run)


def return_matrix(columns: dict[str, list[float] | np.ndarray],
                  start: str = "2020-01-01") -> pd.DataFrame:
    lengths = {len(v) for v in columns.values()}
    assert len(lengths) == 1, "columns must share a length"
    idx = pd.date_range(start, periods=lengths.pop(), freq="B", tz="UTC")
    return pd.DataFrame(columns, index=idx)


FAMILY_VOL = 0.01
FAMILY_CORR = 0.97

SCALE_PAIRED_SHAPES = {
    "a": [0.01, -0.005, 0.02, 0.0, 0.015, -0.01, 0.005, 0.01],
    "b": [-0.01, 0.02, 0.005, 0.01, -0.005, 0.03, 0.0, -0.02],
    "c": [0.005, 0.005, -0.02, 0.01, 0.02, -0.015, 0.01, 0.0],
}


def near_duplicate_family(rng: np.random.Generator, annual_sharpe: float,
                          t: int = 600, n_cols: int = 10) -> pd.DataFrame:
    """One shared signal plus per-column jitter, pairwise correlation
    FAMILY_CORR. Every member has near-identical true skill, so only the jitter
    breaks the in-sample tie — the shape an overlapping parameter window
    produces."""
    jitter = FAMILY_VOL * math.sqrt((1.0 - FAMILY_CORR) / FAMILY_CORR)
    base = rng.normal(annual_sharpe * FAMILY_VOL / math.sqrt(252.0), FAMILY_VOL, t)
    columns = base[:, None] + rng.normal(0.0, jitter, (t, n_cols))
    return return_matrix({f"v{i}": columns[:, i] for i in range(n_cols)})


def drifting_field(drift: float, n_cols: int = 6, t: int = 240,
                   seed: int = 5) -> pd.DataFrame:
    """Every column drifts alike at twenty times its own noise, so the sign of
    any column's Sharpe is the sign of the drift on every split."""
    rng = np.random.default_rng(seed)
    values = drift + 0.001 * rng.standard_normal((t, n_cols))
    return return_matrix({f"c{i}": values[:, i] for i in range(n_cols)})


def one_bump_per_block(n_blocks: int = 8, block_size: int = 10,
                       seed: int = 0) -> pd.DataFrame:
    """Column j earns its whole edge inside block j and bleeds everywhere else,
    so the in-sample winner is always a column whose bump the in-sample legs
    just consumed and whose out-of-sample legs are all losses."""
    rng = np.random.default_rng(seed)
    t = n_blocks * block_size
    columns = {}
    for j in range(n_blocks):
        values = -0.01 + 0.001 * rng.standard_normal(t)
        values[j * block_size:(j + 1) * block_size] += 0.06
        columns[f"blk{j}"] = values
    return return_matrix(columns)


def scale_paired_blocks(shapes: dict[str, list[float]], n_blocks: int = 4,
                        scale: float = 2.0) -> pd.DataFrame:
    """Odd blocks repeat the even blocks scaled up. Sharpe ignores scale, so
    every split's out-of-sample Sharpe reproduces its in-sample Sharpe; a scale
    of 1.0 makes every split score identically."""
    columns = {}
    for label, shape in shapes.items():
        block = np.asarray(shape, dtype=float)
        columns[label] = np.concatenate(
            [block if b % 2 == 0 else scale * block for b in range(n_blocks)]
        )
    return return_matrix(columns)
