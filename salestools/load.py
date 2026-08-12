from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from salestools.core import SalesFrame

_FREQ_MAP = {"D": 1, "W": 7, "M": 30, "Q": 91, "Y": 365}  # approx days
_MEDIAN_TO_FREQ = [(1.5, "D"), (10, "W"), (35, "MS"), (100, "QS"), (400, "YS")]

# Canonical pandas freq strings (avoids deprecated single-letter aliases in pandas 2.2+).
# Weekly is deliberately absent here — pandas anchored-weekly freqs are named after the
# week's last day (e.g. "W-SUN"), and hardcoding one anchor (as an earlier version of this
# code did with "W-MON") silently corrupts reindexing for any data anchored to a different
# weekday: dates that don't fall on the assumed anchor never match the regenerated full
# index, so every period looks "missing" and gets an extra NaN filler row alongside the
# real one. See _weekly_anchor, which detects the anchor from the data instead.
_FREQ_CANONICAL = {"M": "MS", "Q": "QS", "Y": "YS", "D": "D"}

_DOW_ABBR = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]


def _weekly_anchor(dates: pd.DatetimeIndex) -> str:
    """Pick the pandas anchored-weekly freq (e.g. "W-SUN") matching the data's own
    weekday, so reindexing lines up with observed dates instead of assuming Monday."""
    dow = dates.dayofweek.value_counts().idxmax()
    return f"W-{_DOW_ABBR[dow]}"


def _infer_freq(dates: pd.DatetimeIndex) -> str:
    if len(dates) < 2:
        return "D"
    deltas = dates.to_series().diff().dropna().dt.days
    median_gap = deltas.median()
    for threshold, freq in _MEDIAN_TO_FREQ:
        if median_gap <= threshold:
            return freq
    return "Y"


def _fill_gaps(
    group: pd.DataFrame, date_col: str, value_col: str, full_index: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.Series]:
    """Add NaN placeholder rows for dates in `full_index` missing from `group`,
    then forward-fill gaps of <=3 consecutive periods.

    Existing rows are kept as-is, including multiple rows that share a date —
    row-count-sensitive downstream code (e.g. cohort_analysis, which counts rows
    per period) would be corrupted by aggregating them away.
    """
    existing = pd.DatetimeIndex(group[date_col].unique())
    missing = full_index.difference(existing)
    if len(missing) > 0:
        filler = pd.DataFrame({date_col: missing, value_col: np.nan})
        group = pd.concat([group[[date_col, value_col]], filler], ignore_index=True)
    else:
        group = group[[date_col, value_col]].copy()
    group = group.sort_values(date_col).reset_index(drop=True)

    gap_flags = group[value_col].isna()

    # Forward-fill gaps of ≤ 3 consecutive periods; leave longer gaps as NaN
    consecutive = gap_flags.groupby((~gap_flags).cumsum()).cumsum()
    short_gaps = gap_flags & (consecutive <= 3)
    group[value_col] = group[value_col].where(~short_gaps).ffill().where(short_gaps | ~gap_flags, group[value_col])
    group[value_col] = group[value_col].ffill().where(short_gaps, group[value_col])

    # Re-compute: still-missing positions are long gaps (> 3 periods)
    remaining_na = group[value_col].isna()
    gap_flags = short_gaps | remaining_na
    return group, gap_flags


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

    has_segment = bool(segment_col) and segment_col in df.columns

    # Infer freq from the distinct dates present — with a segment column, the
    # same date can appear once per segment (or many times, for per-transaction
    # data), which would corrupt a diff-based median-gap estimate.
    unique_dates = pd.DatetimeIndex(df[date_col].drop_duplicates().sort_values())
    raw_freq = freq or _infer_freq(unique_dates)
    if raw_freq == "W":
        detected_freq = _weekly_anchor(unique_dates)
    else:
        detected_freq = _FREQ_CANONICAL.get(raw_freq, raw_freq)

    if has_segment:
        # Reindex each segment independently, within its own observed date
        # range (not the dataset-wide range), so that (a) multiple rows
        # sharing a date within or across segments don't collide, (b)
        # short-gap forward-fill doesn't bleed values across segment
        # boundaries, and (c) a segment that only exists for part of the
        # overall range (e.g. a cohort that starts later than others) isn't
        # backfilled with NaN rows before it began.
        parts = []
        for seg_value, group in df.groupby(segment_col):
            seg_start, seg_end = group[date_col].min(), group[date_col].max()
            seg_full_index = pd.date_range(seg_start, seg_end, freq=detected_freq)
            filled, gaps = _fill_gaps(group, date_col, value_col, seg_full_index)
            filled[segment_col] = seg_value
            filled["_gap"] = gaps.values
            parts.append(filled)
        combined = pd.concat(parts, ignore_index=True).sort_values(date_col).reset_index(drop=True)
        df = combined.set_index(date_col)
        df.index = pd.DatetimeIndex(df.index)
        gap_flags = df.pop("_gap")
    else:
        start, end = df[date_col].min(), df[date_col].max()
        full_index = pd.date_range(start, end, freq=detected_freq)
        filled, gap_flags = _fill_gaps(df, date_col, value_col, full_index)
        df = filled.set_index(date_col)
        df.index = pd.DatetimeIndex(df.index)
        gap_flags.index = df.index

    if len(df) < 4:
        raise ValueError(
            f"Series has fewer than 4 periods after regularisation ({len(df)}). "
            "Provide more data."
        )

    cols = [value_col]
    if has_segment:
        cols.append(segment_col)

    return SalesFrame(
        data=df[cols].copy(),
        date_col=date_col,
        value_col=value_col,
        freq=detected_freq,
        segment_col=segment_col if has_segment else None,
        gap_flags=gap_flags,
    )
