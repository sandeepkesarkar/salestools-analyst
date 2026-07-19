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


def test_explicit_freq_overrides_inference(tmp_path):
    # Provide daily data but force monthly freq — will be regularised as monthly
    dates = pd.date_range("2022-01-01", periods=12, freq="MS")
    df = pd.DataFrame({"date": dates, "amount": 100.0})
    p = tmp_path / "explicit.csv"
    df.to_csv(p, index=False)
    sf = load_sales(p, freq="MS")
    assert sf.freq == "MS"
