from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from salestools.core import SalesFrame


@dataclass
class CohortTable:
    retention: pd.DataFrame   # cohort × period_offset, values = retention rate 0–1
    fig: plt.Figure = field(default=None)


def cohort_analysis(sf: SalesFrame, cohort_col: str) -> CohortTable:
    """Compute cohort retention from a SalesFrame.

    Groups rows by the first period in which each `cohort_col` value appears,
    then computes period-over-period retention (fraction of cohort still active).

    The SalesFrame must have a `segment_col` or explicit multi-row structure where
    `cohort_col` is a column in `sf.data` (e.g., customer_id or cohort_label).
    """
    df = sf.data.copy()
    df.index.name = sf.date_col

    if cohort_col not in df.columns:
        raise ValueError(
            f"cohort_col '{cohort_col}' not found. Available columns: {list(df.columns)}"
        )

    df = df.reset_index()
    df[sf.date_col] = pd.to_datetime(df[sf.date_col])

    # First period each cohort_col value appears
    cohort_start = df.groupby(cohort_col)[sf.date_col].min().rename("cohort_start")
    df = df.join(cohort_start, on=cohort_col)

    # Period offset (integer)
    df["period_offset"] = (
        (df[sf.date_col] - df["cohort_start"])
        .dt.days // max(1, _median_gap_days(df[sf.date_col]))
    ).astype(int)

    # Cohort size = count at period 0
    cohort_sizes = df[df["period_offset"] == 0].groupby("cohort_start")[cohort_col].count()

    # Active count per (cohort, period)
    active = df.groupby(["cohort_start", "period_offset"])[cohort_col].count().unstack(fill_value=0)

    # Retention = active / cohort_size
    retention = active.div(cohort_sizes, axis=0).fillna(0).clip(upper=1.0)

    fig = _plot_retention(retention)
    return CohortTable(retention=retention, fig=fig)


def _median_gap_days(dates: pd.Series) -> int:
    # Must dedupe before diffing: cohort data has multiple rows per calendar period
    # (one per active cohort member), so diffing the raw per-row column is dominated
    # by same-date (0-day) gaps between rows and collapses the median toward 0 —
    # floored to 1, which turns period_offset into a raw day-count instead of an
    # actual period count (e.g. weekly cohorts end up labeled "P7", "P14", ... instead
    # of "P1", "P2", ...).
    diffs = dates.drop_duplicates().sort_values().diff().dropna().dt.days
    return max(1, int(diffs.median())) if len(diffs) else 1


def _plot_retention(retention: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6, len(retention.columns)), max(4, len(retention) * 0.6)))
    cmap = plt.get_cmap("YlGn")
    im = ax.imshow(retention.values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(retention.columns)))
    ax.set_xticklabels([f"P{c}" for c in retention.columns], fontsize=8)
    ax.set_yticks(range(len(retention.index)))
    ax.set_yticklabels(
        [str(d)[:10] if hasattr(d, "strftime") else str(d) for d in retention.index],
        fontsize=8,
    )
    ax.set_xlabel("Period offset")
    ax.set_ylabel("Cohort start")
    ax.set_title("Cohort Retention")

    for i in range(len(retention.index)):
        for j in range(len(retention.columns)):
            val = retention.iloc[i, j]
            color = "black" if val > 0.5 else "white"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Retention rate")
    plt.tight_layout()
    return fig
