from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from salestools.core import SalesFrame, SegmentRanking


def compare_segments(
    sf: SalesFrame,
    by: Optional[str] = None,
    ranked_by: Literal["cagr", "latest_value"] = "cagr",
) -> SegmentRanking:
    seg_col = by or sf.segment_col
    if not seg_col:
        raise ValueError(
            "No segment column available. Pass `by='column_name'` or load data with `segment_col` set."
        )
    if seg_col not in sf.data.columns:
        raise ValueError(f"Segment column '{seg_col}' not found in SalesFrame.data.")

    freq_periods = {"D": 365, "W": 52, "M": 12, "Q": 4, "Y": 1}
    periods_per_year = freq_periods.get(sf.freq, 365)

    rows = []
    for segment, group in sf.data.groupby(seg_col):
        vals = group[sf.value_col].dropna()
        if len(vals) < 2:
            continue

        n = len(vals)
        years = n / periods_per_year
        if vals.iloc[0] > 0 and years > 0:
            cagr = float((vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1)
        else:
            cagr = float("nan")

        latest = float(vals.iloc[-1])

        # Trend direction from linear slope on last half of series
        half = vals.iloc[len(vals) // 2 :]
        if len(half) >= 2:
            slope = np.polyfit(range(len(half)), half.values, 1)[0]
            if slope > 0.01 * half.mean():
                direction = "up"
            elif slope < -0.01 * half.mean():
                direction = "down"
            else:
                direction = "flat"
        else:
            direction = "flat"

        rows.append({"segment": segment, "cagr": cagr, "latest_value": latest, "trend_direction": direction})

    summary = pd.DataFrame(rows)
    if summary.empty:
        return SegmentRanking(summary=summary, ranked_by=ranked_by)

    summary = summary.sort_values(ranked_by, ascending=False).reset_index(drop=True)
    return SegmentRanking(summary=summary, ranked_by=ranked_by)
