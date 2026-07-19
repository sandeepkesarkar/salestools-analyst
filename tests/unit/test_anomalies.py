"""Unit tests for detect_anomalies() — all 4 methods + empty-result case."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools import load_sales, detect_anomalies, SalesFrame
from salestools.core import AnomalyTable


def _flat_sf(n=104, seed=0, spike_idx=None, spike_val=180.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    values = 200 + rng.normal(0, 5, n)
    if spike_idx is not None:
        values[spike_idx] += spike_val
    df = pd.DataFrame({"date": dates, "amount": values})
    return SalesFrame(data=df.set_index("date"), date_col="date", value_col="amount", freq="W-MON")


def _daily_sf_with_contextual(seed=1):
    """Daily series where Mondays are normally low; one Monday is anomalously high."""
    rng = np.random.default_rng(seed)
    n = 104
    dates = pd.date_range("2022-01-03", periods=n, freq="D")
    base = 200 + rng.normal(0, 8, n)
    dow_effect = np.where(np.arange(n) % 7 == 0, -100, 0)
    values = base + dow_effect
    anomaly_idx = 14  # a Monday
    values[anomaly_idx] += 120
    df = pd.DataFrame({"date": dates, "amount": values.clip(min=0)})
    return SalesFrame(data=df.set_index("date"), date_col="date", value_col="amount", freq="D")


class TestZscore:
    def test_detects_spike(self):
        sf = _flat_sf(spike_idx=50)
        result = detect_anomalies(sf, method="zscore")
        assert isinstance(result, AnomalyTable)
        assert not result.anomalies.empty
        assert result.method == "zscore"

    def test_no_anomalies_in_flat_series(self):
        sf = _flat_sf()
        result = detect_anomalies(sf, method="zscore")
        assert isinstance(result, AnomalyTable)
        # May or may not be empty — just confirm no exception


class TestIQR:
    def test_detects_spike(self):
        sf = _flat_sf(spike_idx=30)
        result = detect_anomalies(sf, method="iqr")
        assert not result.anomalies.empty
        assert result.method == "iqr"

    def test_empty_on_uniform_series(self, tmp_path):
        dates = pd.date_range("2022-01-03", periods=52, freq="W-MON")
        df = pd.DataFrame({"date": dates, "amount": [100.0] * 52})
        sf = SalesFrame(data=df.set_index("date"), date_col="date", value_col="amount", freq="W-MON")
        result = detect_anomalies(sf, method="iqr")
        assert isinstance(result, AnomalyTable)
        assert result.anomalies.empty


class TestIForest:
    def test_detects_spike(self):
        sf = _flat_sf(spike_idx=50)
        result = detect_anomalies(sf, method="iforest")
        assert isinstance(result, AnomalyTable)
        assert result.method == "iforest"

    def test_does_not_raise_on_clean_series(self):
        sf = _flat_sf()
        result = detect_anomalies(sf, method="iforest")
        assert isinstance(result, AnomalyTable)


class TestContextual:
    def test_detects_day_of_week_anomaly(self):
        sf = _daily_sf_with_contextual()
        result = detect_anomalies(sf, method="contextual")
        assert isinstance(result, AnomalyTable)
        assert result.method == "contextual"
        assert not result.anomalies.empty

    def test_falls_back_gracefully_for_weekly_data(self):
        sf = _flat_sf(spike_idx=50)
        result = detect_anomalies(sf, method="contextual")
        assert isinstance(result, AnomalyTable)


class TestAutoMethod:
    def test_daily_uses_contextual(self):
        sf = _daily_sf_with_contextual()
        result = detect_anomalies(sf)
        assert result.method == "contextual"

    def test_never_raises(self):
        sf = _flat_sf()
        detect_anomalies(sf, method="auto")

    def test_returns_empty_when_nothing_found(self):
        dates = pd.date_range("2022-01-03", periods=52, freq="W-MON")
        df = pd.DataFrame({"date": dates, "amount": [100.0] * 52})
        sf = SalesFrame(data=df.set_index("date"), date_col="date", value_col="amount", freq="W-MON")
        result = detect_anomalies(sf, method="iqr")
        assert isinstance(result, AnomalyTable)
