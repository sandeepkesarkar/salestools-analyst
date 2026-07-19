from __future__ import annotations

from salestools.core import AnomalyTable, DecompositionResult, GrowthMetrics, SegmentRanking


def narrate(results) -> None:
    # Import v2 types lazily to avoid circular import at module load time
    from salestools.forecast import ForecastResult
    from salestools.cohort import CohortTable

    if isinstance(results, AnomalyTable):
        _narrate_anomalies(results)
    elif isinstance(results, GrowthMetrics):
        _narrate_growth(results)
    elif isinstance(results, SegmentRanking):
        _narrate_segments(results)
    elif isinstance(results, DecompositionResult):
        _narrate_decomposition(results)
    elif isinstance(results, ForecastResult):
        _narrate_forecast(results)
    elif isinstance(results, CohortTable):
        _narrate_cohort(results)
    else:
        print("No notable findings.")


def _narrate_anomalies(at: AnomalyTable) -> None:
    if at.anomalies.empty:
        print("No notable findings. No anomalies detected in this period.")
        return
    n = len(at.anomalies)
    print(f"Found {n} anomalous period{'s' if n > 1 else ''}:")
    for _, row in at.anomalies.iterrows():
        import pandas as pd
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        print(f"  • {date_str}: {row['label']} (value={row['value']:.1f})")


def _narrate_growth(gm: GrowthMetrics) -> None:
    import math
    if math.isnan(gm.cagr):
        print("No notable findings. Could not compute growth rate.")
        return
    direction = "upward" if gm.cagr > 0 else "downward"
    print(
        f"Overall trend: {direction} — compound annual growth rate of {gm.cagr:.1%}. "
        f"Rolling {gm.window}-period growth tracked {len(gm.rolling_growth)} data points. "
        f"Found {len(gm.inflection_points)} inflection point(s)."
    )


def _narrate_segments(sr: SegmentRanking) -> None:
    if sr.summary.empty:
        print("No notable findings. No segments found.")
        return
    top = sr.summary.iloc[0]
    bottom = sr.summary.iloc[-1]
    print(
        f"Top performer: {top['segment']} (CAGR {top['cagr']:.1%}, trend: {top['trend_direction']}). "
        f"Bottom performer: {bottom['segment']} (CAGR {bottom['cagr']:.1%}, trend: {bottom['trend_direction']})."
    )


def _narrate_decomposition(dr: DecompositionResult) -> None:
    trend_end = float(dr.trend.dropna().iloc[-1]) if not dr.trend.dropna().empty else 0
    trend_start = float(dr.trend.dropna().iloc[0]) if not dr.trend.dropna().empty else 0
    direction = "upward" if trend_end > trend_start else "downward" if trend_end < trend_start else "flat"
    seasonal_strength = float(dr.seasonal.std()) if not dr.seasonal.empty else 0
    print(
        f"Underlying trend: {direction} over the full period. "
        f"Seasonal component strength (std): {seasonal_strength:.2f}. "
        f"Decomposition period: {dr.period}."
    )


def _narrate_forecast(fr) -> None:
    n = len(fr.forecast_series)
    first_val = float(fr.forecast_series.iloc[0]) if n else float("nan")
    last_val = float(fr.forecast_series.iloc[-1]) if n else float("nan")
    direction = "growing" if last_val > first_val else "declining" if last_val < first_val else "flat"
    print(
        f"Forecast: {n} period(s) ahead. "
        f"Projected range: {first_val:.1f} → {last_val:.1f} ({direction}). "
        f"80% confidence interval included."
    )


def _narrate_cohort(ct) -> None:
    if ct.retention.empty:
        print("No notable findings. Cohort table is empty.")
        return
    n_cohorts = len(ct.retention)
    max_periods = ct.retention.shape[1]
    # Period-0 retention should be 1.0 for all; check period-1 avg
    if max_periods >= 2:
        p1_avg = float(ct.retention.iloc[:, 1].mean())
        trend = "strong" if p1_avg >= 0.70 else "moderate" if p1_avg >= 0.40 else "weak"
        print(
            f"Cohort analysis: {n_cohorts} cohort(s) over up to {max_periods} period(s). "
            f"Average period-1 retention: {p1_avg:.0%} ({trend})."
        )
    else:
        print(f"Cohort analysis: {n_cohorts} cohort(s), only baseline period available.")
