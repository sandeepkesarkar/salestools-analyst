from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SalesFrame:
    data: pd.DataFrame          # DatetimeIndex, single value column
    date_col: str
    value_col: str
    freq: str                   # 'D' | 'W' | 'M' | 'Q' | 'Y'
    segment_col: Optional[str] = None
    gap_flags: pd.Series = field(default_factory=pd.Series)


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
