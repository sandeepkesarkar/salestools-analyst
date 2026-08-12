from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SalesFrame:
    data: pd.DataFrame          # DatetimeIndex, single value column
    date_col: str
    value_col: str
    freq: str                   # canonical pandas freq: 'D' | 'W-<anchor>' (e.g. 'W-SUN') | 'MS' | 'QS' | 'YS'
    segment_col: Optional[str] = None
    gap_flags: pd.Series = field(default_factory=pd.Series)


# Weekly freqs are anchored to whichever weekday the source data uses (e.g. "W-SUN",
# "W-MON" — see salestools/load.py's _weekly_anchor), so every consumer of SalesFrame.freq
# that needs an approximate periods-per-year or seasonal-cycle length must match on the
# "W" prefix rather than a single hardcoded anchor string, or it'll silently fall through
# to a wrong default for any data anchored to a day other than the one it happened to
# hardcode. These two helpers are the one place that mapping lives.

def periods_per_year(freq: str, default: int = 365) -> int:
    """Approximate number of periods per year for a SalesFrame.freq value — used to
    annualize growth rates (CAGR)."""
    if freq == "D":
        return 365
    if freq.startswith("W"):
        return 52
    return {"MS": 12, "QS": 4, "YS": 1}.get(freq, default)


def seasonal_period(freq: str, default: int = 7) -> int:
    """Approximate natural seasonal-cycle length (in periods) for a SalesFrame.freq
    value — used as the default STL/Holt-Winters seasonal_periods."""
    if freq == "D":
        return 7
    if freq.startswith("W"):
        return 52
    return {"MS": 12, "QS": 4, "YS": 1}.get(freq, default)


def is_sub_monthly(freq: str) -> bool:
    """True for daily/weekly SalesFrame.freq values — used to choose day-of-week
    bucketing (vs. calendar-month bucketing) for contextual anomaly detection."""
    return freq == "D" or freq.startswith("W")


@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int
    fig: Optional[object] = None  # matplotlib Figure


@dataclass
class AnomalyTable:
    anomalies: pd.DataFrame     # columns: date, value, score, method, label
    method: str
    threshold: float


@dataclass
class GrowthMetrics:
    rolling_growth: pd.Series
    cagr: float
    inflection_points: pd.Series
    window: int


@dataclass
class SegmentRanking:
    summary: pd.DataFrame       # columns: segment, cagr, latest_value, trend_direction
    ranked_by: str
