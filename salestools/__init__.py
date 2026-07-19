"""salestools — Sales time-series analysis library."""

__version__ = "2.0.0"

from salestools.load import load_sales
from salestools.analysis import decompose_trend, growth_metrics
from salestools.anomalies import detect_anomalies
from salestools.segments import compare_segments
from salestools.viz import plot_annotated
from salestools.narrate import narrate
from salestools.forecast import forecast, ForecastResult
from salestools.cohort import cohort_analysis, CohortTable
from salestools.core import (
    SalesFrame,
    DecompositionResult,
    AnomalyTable,
    GrowthMetrics,
    SegmentRanking,
)

__all__ = [
    "load_sales",
    "decompose_trend",
    "growth_metrics",
    "detect_anomalies",
    "compare_segments",
    "plot_annotated",
    "narrate",
    "forecast",
    "cohort_analysis",
    "SalesFrame",
    "DecompositionResult",
    "AnomalyTable",
    "GrowthMetrics",
    "SegmentRanking",
    "ForecastResult",
    "CohortTable",
]
