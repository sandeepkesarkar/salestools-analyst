"""Unit tests for forecast() — seasonal-period resolution and short-series fallback."""
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools import SalesFrame
from salestools.forecast import ForecastResult, forecast


def _weekly_sf(n, freq="W-MON", seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq=freq)
    values = 500 + np.linspace(0, 100, n) + rng.normal(0, 5, n)
    df = pd.DataFrame({"date": dates, "amount": values})
    return SalesFrame(data=df.set_index("date"), date_col="date", value_col="amount", freq=freq)


def test_forecast_raises_below_minimum_length():
    sf = _weekly_sf(3)
    try:
        forecast(sf, horizon=4)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_forecast_falls_back_to_non_seasonal_below_two_cycles():
    """Regression test: once seasonal_period() correctly resolves weekly freqs to 52
    (previously only "W-MON" was recognized, defaulting everything else to no
    seasonality), a single year of weekly data — well under Holt-Winters' 2-full-cycle
    minimum for seasonal initialization — must fall back gracefully, not raise."""
    sf = _weekly_sf(52, freq="W-SUN")
    result = forecast(sf, horizon=8)
    assert isinstance(result, ForecastResult)
    assert len(result.forecast_series) == 8


def test_forecast_uses_seasonal_fit_with_two_full_cycles():
    sf = _weekly_sf(110, freq="W-SUN")
    result = forecast(sf, horizon=8)
    assert isinstance(result, ForecastResult)
    assert len(result.forecast_series) == 8
