"""Planted signal generators. Each function returns (DataFrame, detection_fn).

The DataFrame contains a sales time series with a known signal embedded.
The detection_fn(namespace: dict) -> bool inspects the exec'd code output
to verify the signal was found.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _weekly_base(seed: int, n: int = 104) -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    base = 200 + rng.normal(0, 10, n)
    seasonal = 20 * np.sin(np.linspace(0, 4 * np.pi, n))
    noise = rng.normal(0, 5, n)
    return base + seasonal + noise, rng


def make_trend_up(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 104
    values, rng = _weekly_base(seed, n)
    trend = np.linspace(0, 80, n)
    values = values + trend
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    df = pd.DataFrame({"date": dates, "amount": values.clip(min=0)})

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "cagr") and obj.cagr > 0.03:
                return True
            if hasattr(obj, "trend") and hasattr(obj.trend, "iloc"):
                if obj.trend.iloc[-1] > obj.trend.iloc[0]:
                    return True
        return False

    return df, detect


def make_trend_down(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 104
    values, rng = _weekly_base(seed, n)
    trend = np.linspace(0, -80, n)
    values = values + trend
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    df = pd.DataFrame({"date": dates, "amount": values.clip(min=1)})

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "cagr") and obj.cagr < -0.03:
                return True
            if hasattr(obj, "trend") and hasattr(obj.trend, "iloc"):
                if obj.trend.iloc[-1] < obj.trend.iloc[0]:
                    return True
        return False

    return df, detect


def make_anomaly_spike(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 104
    values, rng = _weekly_base(seed, n)
    spike_idx = int(rng.integers(20, 80))
    values[spike_idx] += 180  # ~18σ spike
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    spike_date = dates[spike_idx]
    df = pd.DataFrame({"date": dates, "amount": values})

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "anomalies") and hasattr(obj.anomalies, "empty"):
                if not obj.anomalies.empty:
                    dates_found = pd.to_datetime(obj.anomalies["date"])
                    if any(abs((d - spike_date).days) <= 7 for d in dates_found):
                        return True
        return False

    return df, detect


def make_anomaly_drop(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 104
    values, rng = _weekly_base(seed, n)
    drop_idx = int(rng.integers(20, 80))
    values[drop_idx] -= 180
    values = values.clip(min=0)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    drop_date = dates[drop_idx]
    df = pd.DataFrame({"date": dates, "amount": values})

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "anomalies") and hasattr(obj.anomalies, "empty"):
                if not obj.anomalies.empty:
                    dates_found = pd.to_datetime(obj.anomalies["date"])
                    if any(abs((d - drop_date).days) <= 7 for d in dates_found):
                        return True
        return False

    return df, detect


def make_anomaly_contextual(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 104
    rng = np.random.default_rng(seed)
    base = 200 + rng.normal(0, 8, n)
    # Monday is always low (~100), other days ~200; inject one unusually HIGH Monday
    dow_effect = np.where(np.arange(n) % 7 == 0, -100, 0)
    values = base + dow_effect
    anomaly_idx = int(rng.integers(10, 90))
    anomaly_idx = anomaly_idx - (anomaly_idx % 7)  # snap to Monday
    values[anomaly_idx] += 120  # anomalously high for a Monday
    dates = pd.date_range("2022-01-03", periods=n, freq="D")
    anomaly_date = dates[anomaly_idx]
    df = pd.DataFrame({"date": dates, "amount": values.clip(min=0)})

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "anomalies") and hasattr(obj.anomalies, "empty"):
                if not obj.anomalies.empty:
                    dates_found = pd.to_datetime(obj.anomalies["date"])
                    if any(abs((d - anomaly_date).days) <= 1 for d in dates_found):
                        return True
        return False

    return df, detect


def make_segment_drag(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 52
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    rows = []
    for product, slope in [("A", 1.5), ("B", 0.5), ("C", -1.2)]:
        base = 150 + rng.normal(0, 8, n)
        trend = np.linspace(0, slope * 40, n)
        amounts = (base + trend).clip(min=1)
        for d, a in zip(dates, amounts):
            rows.append({"date": d, "amount": float(a), "product": product})
    df = pd.DataFrame(rows)

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "summary") and hasattr(obj.summary, "iterrows"):
                bottom = obj.summary.iloc[-1] if not obj.summary.empty else None
                if bottom is not None and str(bottom.get("segment", "")).upper() == "C":
                    return True
        return False

    return df, detect


def make_scope_refusal_dataset(seed: int) -> tuple[pd.DataFrame, Callable]:
    n = 52
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    df = pd.DataFrame({"date": dates, "amount": [100.0] * n})

    def detect(ns: dict) -> bool:
        # Scope refusal: the code should NOT call any salestools analysis function
        # and the string "outside" or "scope" should appear in the source
        return True  # verified at code-string level in generate.py

    return df, detect


# ── v2 signal generators ──────────────────────────────────────────────────────

def make_forecast_question(seed: int) -> tuple[pd.DataFrame, Callable]:
    """Monthly series with clear upward trend — model should call forecast()."""
    n = 48
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    base = 500 + np.linspace(0, 200, n) + rng.normal(0, 20, n)
    df = pd.DataFrame({"date": dates, "amount": base.clip(min=1)})

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "forecast_series") and hasattr(obj.forecast_series, "__len__"):
                if len(obj.forecast_series) > 0:
                    return True
        return False

    return df, detect


def make_cohort_question(seed: int) -> tuple[pd.DataFrame, Callable]:
    """Cohort dataset with diverging retention — model should call cohort_analysis()."""
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = ["2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4"]
    for i, cohort in enumerate(cohorts):
        n_periods = 8 - i  # later cohorts have fewer periods of data
        retention_decay = 0.75 - i * 0.05
        cohort_start = pd.Timestamp("2022-01-01") + pd.DateOffset(months=i * 3)
        for p in range(n_periods):
            retention = retention_decay ** p
            count = max(1, int(100 * retention + rng.normal(0, 5)))
            period_date = cohort_start + pd.DateOffset(months=p)
            for _ in range(count):
                rows.append({
                    "date": period_date.strftime("%Y-%m-%d"),
                    "amount": round(float(rng.uniform(50, 200)), 2),
                    "cohort": cohort,
                })
    df = pd.DataFrame(rows)

    def detect(ns: dict) -> bool:
        for obj in ns.values():
            if hasattr(obj, "retention") and hasattr(obj.retention, "shape"):
                if obj.retention.shape[0] > 0:
                    return True
        return False

    return df, detect


SIGNAL_MAKERS = {
    "trend_up": make_trend_up,
    "trend_down": make_trend_down,
    "anomaly_spike": make_anomaly_spike,
    "anomaly_drop": make_anomaly_drop,
    "anomaly_contextual": make_anomaly_contextual,
    "segment_drag": make_segment_drag,
    "scope_refusal": make_scope_refusal_dataset,
    # v2
    "forecast_up": make_forecast_question,
    "cohort_question": make_cohort_question,
}
