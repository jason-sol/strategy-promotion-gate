import pandas as pd
import pytest


@pytest.fixture
def make_bars():
    def _make(closes, volumes=None, spread=1.0):
        closes = list(closes)
        volumes = list(volumes) if volumes is not None else [1000.0] * len(closes)
        return pd.DataFrame(
            {
                "open": closes,
                "high": [c + spread for c in closes],
                "low": [c - spread for c in closes],
                "close": closes,
                "volume": volumes,
            }
        )

    return _make
