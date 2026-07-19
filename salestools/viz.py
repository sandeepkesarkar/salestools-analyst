from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from salestools.core import AnomalyTable, DecompositionResult, SalesFrame


def plot_annotated(
    sf: SalesFrame,
    anomalies: Optional[AnomalyTable] = None,
    trend: Optional[DecompositionResult] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))

    series = sf.data[sf.value_col]
    ax.plot(series.index, series.values, color="steelblue", linewidth=1.2, label="sales")

    if trend is not None and trend.trend is not None:
        ax.plot(trend.trend.index, trend.trend.values, color="orange",
                linewidth=2, linestyle="--", label="trend")

    if anomalies is not None and not anomalies.anomalies.empty:
        adf = anomalies.anomalies
        ax.scatter(
            pd.to_datetime(adf["date"]),
            adf["value"].astype(float),
            color="red", zorder=5, s=60, label="anomaly",
        )
        for _, row in adf.iterrows():
            ax.annotate(
                row["label"],
                xy=(pd.to_datetime(row["date"]), float(row["value"])),
                xytext=(0, 10), textcoords="offset points",
                fontsize=7, color="red", ha="center",
            )

    ax.set_title(title or "Sales — Annotated")
    ax.set_xlabel("Date")
    ax.set_ylabel(sf.value_col)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
