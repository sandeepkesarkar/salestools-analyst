"""Unit tests for salestools.load — gap-fill, freq detection, error paths."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools import load_sales, SalesFrame

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_load_basic_csv():
    sf = load_sales(FIXTURES / "gaps.csv")
    assert isinstance(sf, SalesFrame)
    assert len(sf.data) >= 4


def test_load_infers_weekly_freq(tmp_path):
    dates = pd.date_range("2022-01-03", periods=52, freq="W-MON")
    df = pd.DataFrame({"date": dates, "amount": 100.0})
    p = tmp_path / "weekly.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p)
    assert sf.freq == "W-MON"


def test_load_infers_monthly_freq(tmp_path):
    dates = pd.date_range("2022-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"date": dates, "amount": 100.0})
    p = tmp_path / "monthly.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p)
    assert sf.freq == "MS"


def test_gap_flags_short_gap(tmp_path):
    """Gaps of ≤3 periods are filled; gap_flags marks them as True."""
    dates = list(pd.date_range("2022-01-03", periods=20, freq="W-MON"))
    # Remove weeks 5–7 (3-period gap)
    dates_with_gap = dates[:4] + dates[7:]
    values = [100.0] * len(dates_with_gap)
    df = pd.DataFrame({"date": dates_with_gap, "amount": values})
    p = tmp_path / "short_gap.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p)
    # After reindexing to full weekly, gap_flags should be True at gap positions
    assert sf.gap_flags.any()


def test_raises_value_error_on_short_series():
    with pytest.raises(ValueError, match="fewer than 4 periods"):
        load_sales(FIXTURES / "short.csv")


def test_raises_type_error_on_non_numeric(tmp_path):
    df = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=10, freq="MS"),
                       "amount": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]})
    p = tmp_path / "bad_values.csv"
    df.to_csv(p, index=False)
    with pytest.raises(TypeError):
        load_sales(p)


def test_raises_value_error_on_missing_date_col(tmp_path):
    df = pd.DataFrame({"ts": pd.date_range("2022-01-01", periods=10, freq="MS"),
                       "amount": 100.0})
    p = tmp_path / "no_date.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="date_col"):
        load_sales(p)


def test_segment_col_loaded(tmp_path):
    dates = pd.date_range("2022-01-03", periods=20, freq="W-MON")
    df = pd.DataFrame({"date": dates, "amount": 100.0, "product": "A"})
    p = tmp_path / "seg.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p, segment_col="product")
    assert sf.segment_col == "product"
    assert "product" in sf.data.columns


def test_multi_segment_shared_dates_does_not_crash(tmp_path):
    """Multiple segments sharing the same dates (one row per date per segment)
    must not crash load_sales's reindex (regression: previously raised
    ValueError: cannot reindex on an axis with duplicate labels)."""
    dates = pd.date_range("2022-01-03", periods=20, freq="W-MON")
    rows = []
    for product in ("A", "B", "C"):
        for d in dates:
            rows.append({"date": d, "amount": 100.0, "product": product})
    df = pd.DataFrame(rows)
    p = tmp_path / "multi_seg.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p, segment_col="product")
    assert set(sf.data["product"].unique()) == {"A", "B", "C"}
    assert len(sf.data) == 3 * len(dates)


def test_segment_with_many_rows_per_date_preserves_row_count(tmp_path):
    """Segments with many rows sharing a single date (e.g. per-transaction
    data) must keep every row rather than collapsing them — downstream code
    like cohort_analysis counts rows per period."""
    rows = []
    for cohort, start, n_txns in [("Q1", "2022-01-01", 5), ("Q2", "2022-04-01", 3)]:
        for _ in range(n_txns):
            rows.append({"date": start, "amount": 50.0, "cohort": cohort})
    df = pd.DataFrame(rows)
    p = tmp_path / "cohort_multi.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p, segment_col="cohort")
    assert len(sf.data) == 8  # 5 + 3, none aggregated away
    assert (sf.data["cohort"] == "Q1").sum() == 5
    assert (sf.data["cohort"] == "Q2").sum() == 3


def test_segment_gap_fill_scoped_to_own_date_range(tmp_path):
    """A segment that starts later than others must not be backfilled with
    NaN rows before its own first observed date (regression: earlier fix
    used the dataset-wide date range for every segment, which made every
    segment's earliest date collapse to the same value)."""
    rows = []
    for d in pd.date_range("2022-01-03", periods=8, freq="W-MON"):
        rows.append({"date": d, "amount": 100.0, "cohort": "early"})
    for d in pd.date_range("2022-06-06", periods=8, freq="W-MON"):
        rows.append({"date": d, "amount": 100.0, "cohort": "late"})
    df = pd.DataFrame(rows)
    p = tmp_path / "staggered.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p, segment_col="cohort")
    late_dates = sf.data[sf.data["cohort"] == "late"].index
    assert late_dates.min() == pd.Timestamp("2022-06-06")


def test_explicit_freq_overrides_inference(tmp_path):
    # Provide daily data but force monthly freq — will be regularised as monthly
    dates = pd.date_range("2022-01-01", periods=12, freq="MS")
    df = pd.DataFrame({"date": dates, "amount": 100.0})
    p = tmp_path / "explicit.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p, freq="MS")
    assert sf.freq == "MS"
