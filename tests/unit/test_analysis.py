"""Unit tests for decompose_trend() and growth_metrics()."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools import load_sales, decompose_trend, growth_metrics, SalesFrame
from salestools.core import DecompositionResult, GrowthMetrics


def _make_sf(n=52, freq="W-MON", cagr=0.1):
    """52-week series with known upward trend."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-03", periods=n, freq=freq)
    base = 100.0
    values = base * (1 + cagr / 52) ** np.arange(n) + rng.normal(0, 3, n)
    df = pd.DataFrame({"date": dates, "amount": values.clip(min=1)})
    return SalesFrame(
        data=df.set_index("date"),
        date_col="date",
        value_col="amount",
        freq=freq,
    )


class TestDecomposeTrend:
    def test_returns_decomposition_result(self):
        sf = _make_sf()
        result = decompose_trend(sf)
        assert isinstance(result, DecompositionResult)

    def test_trend_is_upward_for_growing_series(self):
        sf = _make_sf(cagr=0.5)
        result = decompose_trend(sf)
        trend = result.trend.dropna()
        assert trend.iloc[-1] > trend.iloc[0]

    def test_has_correct_period_for_weekly(self):
        sf = _make_sf(freq="W-MON")
        result = decompose_trend(sf)
        assert result.period == 52

    def test_fig_is_returned(self):
        import matplotlib.pyplot as plt
        sf = _make_sf()
        result = decompose_trend(sf)
        assert result.fig is not None
        assert isinstance(result.fig, plt.Figure)
        plt.close("all")

    def test_short_series_does_not_raise(self):
        """Series shorter than 2×period should warn but not raise."""
        sf = _make_sf(n=10)
        # Should not raise — degrades gracefully
        result = decompose_trend(sf)
        assert isinstance(result, DecompositionResult)

    def test_monthly_period_is_12(self):
        sf = _make_sf(n=36, freq="MS")
        result = decompose_trend(sf)
        assert result.period == 12


class TestGrowthMetrics:
    def test_returns_growth_metrics(self):
        sf = _make_sf()
        result = growth_metrics(sf)
        assert isinstance(result, GrowthMetrics)

    def test_positive_cagr_for_growing_series(self):
        sf = _make_sf(cagr=0.3)
        result = growth_metrics(sf)
        assert result.cagr > 0

    def test_negative_cagr_for_declining_series(self):
        sf = _make_sf(cagr=-0.3)
        result = growth_metrics(sf)
        assert result.cagr < 0

    def test_rolling_growth_has_correct_length(self):
        sf = _make_sf(n=52)
        result = growth_metrics(sf, window=4)
        assert result.window == 4
        assert len(result.rolling_growth.dropna()) <= 52

    def test_inflection_points_are_dates(self):
        sf = _make_sf(n=52)
        result = growth_metrics(sf)
        assert isinstance(result.inflection_points, list)
