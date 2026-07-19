"""Unit tests for narrate() — all result types + empty-result case."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools import narrate
from salestools.core import (
    AnomalyTable, GrowthMetrics, SegmentRanking, DecompositionResult
)
from salestools.forecast import ForecastResult
from salestools.cohort import CohortTable


def _make_anomaly_table(empty=False):
    if empty:
        return AnomalyTable(anomalies=pd.DataFrame(columns=["date", "value", "label"]),
                            method="zscore", threshold=3.0)
    df = pd.DataFrame({
        "date": [pd.Timestamp("2022-06-06")],
        "value": [450.0],
        "label": ["spike"],
    })
    return AnomalyTable(anomalies=df, method="zscore", threshold=3.0)


def _make_growth_metrics():
    dates = pd.date_range("2022-01-03", periods=52, freq="W-MON")
    rolling = pd.Series([0.01] * 48, index=dates[4:])
    return GrowthMetrics(
        rolling_growth=rolling,
        cagr=0.12,
        inflection_points=[dates[20]],
        window=4,
    )


def _make_segment_ranking():
    df = pd.DataFrame({
        "segment": ["A", "B", "C"],
        "cagr": [0.15, 0.02, -0.08],
        "latest_value": [200.0, 150.0, 80.0],
        "trend_direction": ["up", "flat", "down"],
    })
    return SegmentRanking(summary=df, ranked_by="cagr")


def _make_decomposition():
    dates = pd.date_range("2022-01-03", periods=52, freq="W-MON")
    return DecompositionResult(
        trend=pd.Series(np.linspace(100, 200, 52), index=dates),
        seasonal=pd.Series(np.sin(np.linspace(0, 4*np.pi, 52)) * 10, index=dates),
        residual=pd.Series(np.zeros(52), index=dates),
        period=52,
        fig=None,
    )


def _make_forecast_result():
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    forecast_vals = pd.Series(np.linspace(100, 150, 12), index=dates)
    ci = pd.DataFrame({"lower": forecast_vals - 10, "upper": forecast_vals + 10})
    return ForecastResult(forecast_series=forecast_vals, confidence_interval=ci)


def _make_cohort_table():
    retention = pd.DataFrame(
        [[1.0, 0.75, 0.60], [1.0, 0.65, 0.50]],
        index=[pd.Timestamp("2022-01-01"), pd.Timestamp("2022-04-01")],
        columns=[0, 1, 2],
    )
    return CohortTable(retention=retention)


class TestNarrateAnomalyTable:
    def test_narrates_anomalies(self, capsys):
        narrate(_make_anomaly_table())
        out = capsys.readouterr().out
        assert "2022-06-06" in out

    def test_no_findings_on_empty(self, capsys):
        narrate(_make_anomaly_table(empty=True))
        out = capsys.readouterr().out
        assert "No notable findings" in out


class TestNarrateGrowthMetrics:
    def test_shows_cagr(self, capsys):
        narrate(_make_growth_metrics())
        out = capsys.readouterr().out
        assert "12" in out or "CAGR" in out.upper() or "growth" in out.lower()

    def test_shows_direction(self, capsys):
        narrate(_make_growth_metrics())
        out = capsys.readouterr().out
        assert "upward" in out or "up" in out.lower()


class TestNarrateSegmentRanking:
    def test_shows_top_segment(self, capsys):
        narrate(_make_segment_ranking())
        out = capsys.readouterr().out
        assert "A" in out

    def test_shows_bottom_segment(self, capsys):
        narrate(_make_segment_ranking())
        out = capsys.readouterr().out
        assert "C" in out


class TestNarrateDecompositionResult:
    def test_shows_trend_direction(self, capsys):
        narrate(_make_decomposition())
        out = capsys.readouterr().out
        assert "upward" in out or "trend" in out.lower()


class TestNarrateForecastResult:
    def test_shows_forecast_periods(self, capsys):
        narrate(_make_forecast_result())
        out = capsys.readouterr().out
        assert "12" in out or "Forecast" in out or "forecast" in out.lower()


class TestNarrateCohortTable:
    def test_shows_cohort_count(self, capsys):
        narrate(_make_cohort_table())
        out = capsys.readouterr().out
        assert "2" in out or "cohort" in out.lower()

    def test_empty_cohort_shows_no_findings(self, capsys):
        narrate(CohortTable(retention=pd.DataFrame()))
        out = capsys.readouterr().out
        assert "No notable findings" in out


class TestNarrateUnknownType:
    def test_prints_no_findings_for_unknown(self, capsys):
        narrate("not a result type")
        out = capsys.readouterr().out
        assert "No notable findings" in out

    def test_never_raises(self):
        narrate(None)
        narrate(42)
        narrate([])
