from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from salestools.core import DecompositionResult, GrowthMetrics, SalesFrame, periods_per_year, seasonal_period


def decompose_trend(
    sf: SalesFrame,
    period: int | str = "auto",
) -> DecompositionResult:
    series = sf.data[sf.value_col].dropna()

    if period == "auto":
        resolved_period = seasonal_period(sf.freq)
    else:
        resolved_period = int(period)

    min_length = 2 * resolved_period
    too_short = len(series) < min_length

    if too_short or resolved_period <= 1:
        print(
            f"Warning: series length ({len(series)}) is less than 2 × period ({resolved_period}). "
            "Returning trend-only decomposition (no seasonal component)."
        )
        trend = series.rolling(window=max(3, resolved_period // 2), center=True, min_periods=1).mean()
        seasonal = pd.Series(0.0, index=series.index)
        residual = series - trend
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(series.index, trend, label="trend")
        ax.set_title("Trend (seasonal decomposition unavailable — too few periods)")
        ax.legend()
        return DecompositionResult(
            trend=trend, seasonal=seasonal, residual=residual, period=resolved_period, fig=fig
        )

    stl = STL(series, period=resolved_period, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(series.index, series, label="observed")
    axes[0].set_ylabel("Observed")
    axes[1].plot(series.index, result.trend, label="trend", color="C1")
    axes[1].set_ylabel("Trend")
    axes[2].plot(series.index, result.seasonal, label="seasonal", color="C2")
    axes[2].set_ylabel("Seasonal")
    axes[3].plot(series.index, result.resid, label="residual", color="C3")
    axes[3].set_ylabel("Residual")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("STL Decomposition", y=1.01)
    plt.tight_layout()

    return DecompositionResult(
        trend=result.trend,
        seasonal=result.seasonal,
        residual=result.resid,
        period=resolved_period,
        fig=fig,
    )


def growth_metrics(sf: SalesFrame, window: int = 4) -> GrowthMetrics:
    series = sf.data[sf.value_col].dropna()

    rolling_growth = series.pct_change(periods=window).dropna()

    n_periods = len(series)
    if n_periods >= 2 and series.iloc[0] > 0:
        years = n_periods / periods_per_year(sf.freq)
        cagr = float((series.iloc[-1] / series.iloc[0]) ** (1 / max(years, 0.01)) - 1)
    else:
        cagr = float("nan")

    # A true inflection point is where rolling_growth actually crosses zero — current
    # and previous values have opposite sign, so their product is negative. (An earlier
    # version compared the *derivative* of rolling_growth against the previous value
    # instead of comparing consecutive values' signs directly, which fired on every step
    # adjacent to a real crossing too — e.g. growth moving -2 → -1 → 3 flagged both -1
    # and 3 as inflections, even though -1 never actually crossed zero.)
    inflections = rolling_growth[(rolling_growth * rolling_growth.shift(1)) < 0].index

    return GrowthMetrics(
        rolling_growth=rolling_growth,
        cagr=cagr,
        inflection_points=list(inflections),
        window=window,
    )
