from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from salestools.core import SalesFrame


@dataclass
class ForecastResult:
    forecast_series: pd.Series
    confidence_interval: pd.DataFrame
    fig: plt.Figure = field(default=None)


def forecast(sf: SalesFrame, horizon: int = 12) -> ForecastResult:
    """Holt-Winters exponential smoothing forecast.

    Returns `horizon` future periods with 80% confidence interval.
    """
    series = sf.data[sf.value_col].dropna()

    if len(series) < 4:
        raise ValueError(f"Need at least 4 observations to forecast, got {len(series)}.")

    # Choose seasonal periods based on freq
    _PERIOD = {"D": 7, "W-MON": 52, "MS": 12, "QS": 4, "YS": 1}
    seasonal_periods = _PERIOD.get(sf.freq, 1)

    trend_type = "add"
    seasonal_type = "add" if seasonal_periods > 1 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            series,
            trend=trend_type,
            seasonal=seasonal_type,
            seasonal_periods=seasonal_periods if seasonal_type else None,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)

    forecast_vals = model.forecast(horizon)

    # Build approximate 80% CI using in-sample residual std
    resid_std = model.resid.std()
    z = 1.28  # 80% CI
    lower = forecast_vals - z * resid_std
    upper = forecast_vals + z * resid_std
    ci = pd.DataFrame({"lower": lower, "upper": upper})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, label="Actual", color="steelblue")
    ax.plot(forecast_vals.index, forecast_vals.values, label="Forecast", color="darkorange", linestyle="--")
    ax.fill_between(ci.index, ci["lower"], ci["upper"], alpha=0.2, color="darkorange", label="80% CI")
    ax.set_title(f"Forecast: next {horizon} {sf.freq} periods")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return ForecastResult(
        forecast_series=forecast_vals,
        confidence_interval=ci,
        fig=fig,
    )
