"""Unit tests for compare_segments() ranking and error on missing segment column."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools import load_sales, compare_segments, SalesFrame
from salestools.core import SegmentRanking


def _multi_segment_sf(seed=0):
    """3-product weekly SF: A grows, B flat, C declines."""
    rng = np.random.default_rng(seed)
    n = 52
    dates = pd.date_range("2022-01-03", periods=n, freq="W-MON")
    rows = []
    for product, slope in [("A", 1.5), ("B", 0.0), ("C", -1.2)]:
        base = 150 + rng.normal(0, 5, n)
        trend = np.linspace(0, slope * 40, n)
        amounts = (base + trend).clip(min=1)
        for d, a in zip(dates, amounts):
            rows.append({"date": d, "amount": float(a), "product": product})
    df = pd.DataFrame(rows)
    return SalesFrame(
        data=df.set_index("date")[["amount", "product"]],
        date_col="date",
        value_col="amount",
        freq="W-MON",
        segment_col="product",
    )


def test_returns_segment_ranking():
    sf = _multi_segment_sf()
    result = compare_segments(sf)
    assert isinstance(result, SegmentRanking)


def test_summary_has_all_segments():
    sf = _multi_segment_sf()
    result = compare_segments(sf)
    segments = set(result.summary["segment"].tolist())
    assert segments == {"A", "B", "C"}


def test_bottom_performer_is_c():
    """Product C has negative slope — should appear last after sorting by CAGR."""
    sf = _multi_segment_sf()
    result = compare_segments(sf)
    bottom = result.summary.iloc[-1]
    assert str(bottom["segment"]).upper() == "C"


def test_top_performer_is_a():
    """Product A has strongest positive slope."""
    sf = _multi_segment_sf()
    result = compare_segments(sf)
    top = result.summary.iloc[0]
    assert str(top["segment"]).upper() == "A"


def test_trend_direction_columns_present():
    sf = _multi_segment_sf()
    result = compare_segments(sf)
    assert "trend_direction" in result.summary.columns
    assert "cagr" in result.summary.columns


def test_raises_value_error_without_segment_col(tmp_path):
    dates = pd.date_range("2022-01-03", periods=20, freq="W-MON")
    df = pd.DataFrame({"date": dates, "amount": 100.0})
    sf = SalesFrame(
        data=df.set_index("date"),
        date_col="date",
        value_col="amount",
        freq="W-MON",
        segment_col=None,
    )
    with pytest.raises(ValueError, match="segment"):
        compare_segments(sf)
