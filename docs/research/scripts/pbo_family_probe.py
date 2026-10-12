"""Does CSCV/PBO measure overfitting, or family homogeneity?

Five synthetic candidate sets, N=30 columns of daily returns, T=1500 bars each:
  (a) correlated family, true Sharpe 1.0   -- near-duplicate param variants of a real edge
  (b) correlated family, true Sharpe 0     -- near-duplicate variants of nothing
  (c) heterogeneous independent noise      -- 30 unrelated strategies, none has an edge
  (d) heterogeneous + one genuinely good   -- 30 unrelated, one real edge
  (e) correlated family with a true optimum -- the shape a real param grid has

Headline PBO comes from the repo's own bot.validation.cscv_pbo. A vectorised
replica computes the paper's other CSCV outputs (logit distribution, OOS
performance-degradation slope, probability of OOS loss), which the repo scalar
does not expose. Parity between the two is asserted on every scenario/seed.

Run from the repo root:  uv run python docs/research/scripts/pbo_family_probe.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from bot.validation import cscv_pbo  # noqa: E402

T = 1500          # ~6y of daily bars, matches the real runs' window
N = 30
SIG = 0.01        # daily vol
ANN = np.sqrt(252.0)
CORR = 0.97       # target pairwise correlation inside a "family"
N_BLOCKS = 16     # 12870 splits, the repo default
SEEDS = range(20)


def _mu(annual_sharpe: float) -> float:
    return annual_sharpe * SIG / ANN


def family(rng, base_sharpe, per_col_sharpe=None):
    """One base signal + tiny per-column jitter -> pairwise corr ~ CORR."""
    k = SIG * np.sqrt((1.0 - CORR) / CORR)
    base = rng.normal(_mu(base_sharpe), SIG, T)
    cols = base[:, None] + rng.normal(0.0, k, (T, N))
    if per_col_sharpe is not None:
        cols = cols + _mu(np.asarray(per_col_sharpe))[None, :]
    return cols


def heterogeneous(rng, sharpes):
    return np.column_stack([rng.normal(_mu(s), SIG, T) for s in sharpes])


def scenarios(seed):
    rng = np.random.default_rng(seed)
    # (e): smooth hump across the param axis, peak +0.6 annual Sharpe at centre
    hump = 0.6 * np.exp(-0.5 * ((np.arange(N) - (N - 1) / 2) / 6.0) ** 2)
    return {
        "a corr family, true SR 1.0": family(rng, 1.0),
        "b corr family, true SR 0": family(rng, 0.0),
        "c indep noise x30": heterogeneous(rng, np.zeros(N)),
        "d indep noise + 1 good": heterogeneous(rng, [1.5] + [0.0] * (N - 1)),
        "e corr family w/ true optimum": family(rng, 0.3, per_col_sharpe=hump),
    }


def cscv_full(values, n_blocks=N_BLOCKS):
    """Vectorised CSCV. Returns the repo's PBO plus the paper's other outputs."""
    t, n_cols = values.shape
    bs = t // n_blocks
    blocks = values[: bs * n_blocks].reshape(n_blocks, bs, n_cols)
    bsum = blocks.sum(axis=1)                # (n_blocks, n_cols)
    bsq = (blocks ** 2).sum(axis=1)
    half = n_blocks // 2

    combos = np.array([[b in c for b in range(n_blocks)]
                       for c in itertools.combinations(range(n_blocks), half)],
                      dtype=float)
    n_obs = half * bs

    def sharpe(mask):
        s = mask @ bsum
        q = mask @ bsq
        mean = s / n_obs
        var = (q - n_obs * mean ** 2) / (n_obs - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        return np.divide(mean, std, out=np.zeros_like(mean), where=std > 0)

    is_s = sharpe(combos)                    # (n_splits, n_cols)
    oos_s = sharpe(1.0 - combos)
    best = np.argmax(is_s, axis=1)
    rows = np.arange(len(best))
    best_oos = oos_s[rows, best]
    best_is = is_s[rows, best]

    # repo's rank statistic: fraction of the field the IS winner beats OOS
    w = (oos_s < best_oos[:, None]).mean(axis=1)
    pbo = float((w < 0.5).mean())

    # paper's logit; clip only the degenerate endpoints
    wc = np.clip(w, 1.0 / (2 * n_cols), 1.0 - 1.0 / (2 * n_cols))
    logit = np.log(wc / (1.0 - wc))

    # performance degradation: OLS of OOS Sharpe on IS Sharpe for the selected column
    slope = np.polyfit(best_is, best_oos, 1)[0] if best_is.std() > 0 else float("nan")
    return {
        "pbo": pbo,
        "median_logit": float(np.median(logit)),
        "degrade_slope": float(slope),
        "p_oos_loss": float((best_oos <= 0).mean()),
        "mean_oos_sr_ann": float(best_oos.mean() * ANN),
        "mean_is_sr_ann": float(best_is.mean() * ANN),
    }


def dedupe_keep(corr, threshold):
    """Greedy medoid dedupe: keep a column only if it is below threshold
    against everything kept so far."""
    keep = []
    for i in range(len(corr)):
        if all(corr[i, j] < threshold for j in keep):
            keep.append(i)
    return keep


def main():
    rows = []
    for seed in SEEDS:
        for name, vals in scenarios(seed).items():
            frame = pd.DataFrame(vals, columns=[f"c{i}" for i in range(vals.shape[1])])
            repo_pbo = cscv_pbo(frame, n_blocks=N_BLOCKS)
            full = cscv_full(vals, n_blocks=N_BLOCKS)
            assert abs(repo_pbo - full["pbo"]) < 1e-12, (name, seed, repo_pbo, full["pbo"])
            rows.append({"scenario": name, "seed": seed, **full})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220, "display.max_columns", 60)

    print(f"=== Table 1: PBO sampling distribution, {len(list(SEEDS))} seeds "
          f"(repo cscv_pbo, n_blocks={N_BLOCKS}, {len(df)} runs) ===")
    print(df.groupby("scenario")["pbo"].agg(["mean", "std", "min", "max"]).round(3))

    print("\n=== Table 2: the paper's four panels, mean over seeds ===")
    print(df.groupby("scenario")[["pbo", "p_oos_loss", "mean_oos_sr_ann",
                                  "median_logit", "degrade_slope"]].mean().round(3))

    print("\n=== (a) vs (b) per-seed separation: how often does each stat get it right? ===")
    a = df[df.scenario.str.startswith("a")].set_index("seed")
    b = df[df.scenario.str.startswith("b")].set_index("seed")
    print(f"  PBO             lower for the real-edge family in {(a.pbo < b.pbo).mean():.0%} of seeds")
    print(f"  p_oos_loss      lower for the real-edge family in {(a.p_oos_loss < b.p_oos_loss).mean():.0%} of seeds")
    print(f"  mean OOS Sharpe higher for the real-edge family in {(a.mean_oos_sr_ann > b.mean_oos_sr_ann).mean():.0%} of seeds")
    print(f"  p_oos_loss ranges: real-edge {a.p_oos_loss.min():.2f}-{a.p_oos_loss.max():.2f}, "
          f"no-edge {b.p_oos_loss.min():.2f}-{b.p_oos_loss.max():.2f}")

    print("\n=== option A: what a correlation pre-cluster leaves (seed 0) ===")
    corrs = {name: np.corrcoef(vals, rowvar=False) for name, vals in scenarios(0).items()}
    for thr in (0.90, 0.95, 0.99):
        counts = {name[:1]: len(dedupe_keep(c, thr)) for name, c in corrs.items()}
        print(f"  thr {thr}: " + "  ".join(f"{k}={v}/{N}" for k, v in counts.items()))
    print("  PBO over the survivors (nan = fewer than 2 reps, gate passes through):")
    for name, vals in scenarios(0).items():
        keep = dedupe_keep(corrs[name], 0.90)
        p = cscv_pbo(pd.DataFrame(vals[:, keep]), n_blocks=N_BLOCKS) if len(keep) >= 2 else float("nan")
        print(f"    {name:32s} {len(keep):2d}/{N} reps   PBO {p:.3f}")


if __name__ == "__main__":
    main()
