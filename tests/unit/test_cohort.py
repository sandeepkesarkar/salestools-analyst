"""Unit tests for cohort_analysis() — retention rate + period_offset labeling.

Previously had zero test coverage anywhere in the suite (only exercised indirectly
via narrate() tests, which construct a CohortTable dataclass directly rather than
calling cohort_analysis()."""
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from salestools.core import SalesFrame
from salestools.cohort import cohort_analysis, CohortTable


def _cohort_sf():
    """2 cohorts, weekly cadence, multiple customer rows sharing each date —
    exactly the shape that triggers the _median_gap_days duplicate-date bug."""
    dates = pd.date_range("2022-01-03", periods=4, freq="W-MON")
    rows = []
    # Cohort starting week 0: C1 stays active all 4 weeks, C2 churns after week 1.
    for d in dates:
        rows.append({"date": d, "amount": 10.0, "customer_id": "C1"})
    for d in dates[:2]:
        rows.append({"date": d, "amount": 10.0, "customer_id": "C2"})
    # Cohort starting week 1: C3 stays active for its remaining 3 weeks.
    for d in dates[1:]:
        rows.append({"date": d, "amount": 10.0, "customer_id": "C3"})
    df = pd.DataFrame(rows).set_index("date")
    return SalesFrame(data=df, date_col="date", value_col="amount", freq="W-MON"), dates


class TestCohortAnalysis:
    def test_returns_cohort_table(self):
        sf, _ = _cohort_sf()
        result = cohort_analysis(sf, cohort_col="customer_id")
        assert isinstance(result, CohortTable)

    def test_period_offset_columns_are_periods_not_raw_days(self):
        """Regression test: _median_gap_days used to compute the median gap over the
        raw per-row date column, which is dominated by same-date (0-day) gaps when
        multiple cohort members share a date — collapsing the median to 0, floored to
        1, so period_offset ended up counting raw days (0, 7, 14, 21) instead of
        actual periods (0, 1, 2, 3)."""
        sf, _ = _cohort_sf()
        result = cohort_analysis(sf, cohort_col="customer_id")
        assert list(result.retention.columns) == [0, 1, 2, 3]

    def test_retention_rate_for_partial_churn(self):
        sf, dates = _cohort_sf()
        result = cohort_analysis(sf, cohort_col="customer_id")
        cohort0 = result.retention.loc[dates[0]]
        # 2 members at period 0, only C1 remains by period 2 and 3.
        assert cohort0[0] == 1.0
        assert cohort0[1] == 1.0
        assert cohort0[2] == 0.5
        assert cohort0[3] == 0.5

    def test_later_cohort_starts_at_its_own_period_zero(self):
        sf, dates = _cohort_sf()
        result = cohort_analysis(sf, cohort_col="customer_id")
        cohort1 = result.retention.loc[dates[1]]
        assert cohort1[0] == 1.0
        assert cohort1[1] == 1.0
        assert cohort1[2] == 1.0

    def test_raises_on_missing_cohort_col(self):
        sf, _ = _cohort_sf()
        try:
            cohort_analysis(sf, cohort_col="nonexistent")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_fig_is_returned(self):
        import matplotlib.pyplot as plt
        sf, _ = _cohort_sf()
        result = cohort_analysis(sf, cohort_col="customer_id")
        assert result.fig is not None
        assert isinstance(result.fig, plt.Figure)
        plt.close("all")
