from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from salestools.core import SalesFrame

_FREQ_MAP = {"D": 1, "W": 7, "M": 30, "Q": 91, "Y": 365}  # approx days
_MEDIAN_TO_FREQ = [(1.5, "D"), (10, "W-MON"), (35, "MS"), (100, "QS"), (400, "YS")]

# Canonical pandas freq strings (avoids deprecated single-letter aliases in pandas 2.2+)
_FREQ_CANONICAL = {"W": "W-MON", "M": "MS", "Q": "QS", "Y": "YS", "D": "D"}


def _infer_freq(dates: pd.DatetimeIndex) -> str:
    if len(dates) < 2:
        return "D"
    deltas = dates.to_series().diff().dropna().dt.days
    median_gap = deltas.median()
    for threshold, freq in _MEDIAN_TO_FREQ:
        if median_gap <= threshold:
            return freq
    return "Y"


def load_sales(
    path: str | Path,
    date_col: str = "date",
    value_col: str = "amount",
    segment_col: Optional[str] = None,
    freq: Optional[str] = None,
) -> SalesFrame:
    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in CSV columns: {list(df.columns)}")
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in CSV columns: {list(df.columns)}")

    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as exc:
        raise ValueError(f"Cannot parse '{date_col}' as dates: {exc}") from exc

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        try:
            df[value_col] = pd.to_numeric(df[value_col])
        except Exception as exc:
            raise TypeError(f"'{value_col}' cannot be coerced to numeric: {exc}") from exc

    df = df.sort_values(date_col).reset_index(drop=True)

    raw_freq = freq or _infer_freq(pd.DatetimeIndex(df[date_col]))
    detected_freq = _FREQ_CANONICAL.get(raw_freq, raw_freq)

    # Build a full regular index to detect gaps
    start, end = df[date_col].min(), df[date_col].max()
    full_index = pd.date_range(start, end, freq=detected_freq)
    df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(full_index)

    if len(df) < 4:
        raise ValueError(
            f"Series has fewer than 4 periods after regularisation ({len(df)}). "
            "Provide more data."
        )

    gap_flags = df[value_col].isna()

    # Forward-fill gaps of ≤ 3 consecutive periods; leave longer gaps as NaN
    consecutive = gap_flags.groupby((~gap_flags).cumsum()).cumsum()
    short_gaps = gap_flags & (consecutive <= 3)
    df[value_col] = df[value_col].where(~short_gaps).ffill().where(short_gaps | ~gap_flags, df[value_col])
    df[value_col] = df[value_col].ffill().where(short_gaps, df[value_col])

    # Re-compute: still-missing positions are long gaps (> 3 periods)
    remaining_na = df[value_col].isna()
    gap_flags = short_gaps | remaining_na

    cols = [value_col]
    if segment_col and segment_col in df.columns:
        cols.append(segment_col)

    return SalesFrame(
        data=df[cols].copy(),
        date_col=date_col,
        value_col=value_col,
        freq=detected_freq,
        segment_col=segment_col if segment_col in (df.columns if segment_col else []) else None,
        gap_flags=gap_flags,
    )
