# Contract: `salestools` Public Python API

**Version**: 1.0.0 | **Date**: 2026-07-13

This document defines the public surface of the `salestools` library. Any change to a function
signature, return type, or raised exception is a breaking change requiring a semver MAJOR bump.

---

## `load_sales`

```python
def load_sales(
    path: str | Path,
    date_col: str = "date",
    value_col: str = "amount",
    segment_col: Optional[str] = None,
    freq: Optional[str] = None,
) -> SalesFrame
```

**Behaviour**:
- Reads a CSV at `path`; parses `date_col` as datetime, `value_col` as float.
- Detects or accepts `freq` ('D', 'W', 'M', 'Q', 'Y'); infers from median gap if not supplied.
- Fills gaps ≤3 consecutive periods by forward-fill (sets `gap_flags=True` for those rows).
- Raises `ValueError` if date parsing fails, series length < 4, or frequency cannot be inferred.
- Raises `TypeError` if `value_col` is non-numeric.

**Returns**: `SalesFrame`

---

## `decompose_trend`

```python
def decompose_trend(
    sf: SalesFrame,
    period: int | str = "auto",
) -> DecompositionResult
```

**Behaviour**:
- Runs STL decomposition (statsmodels).
- `period="auto"`: D→7, W→52, M→12, Q→4, Y→1.
- If series length < 2×period: prints warning and returns trend-only (no seasonal component).
- Always plots and returns the decomposition figure.

**Returns**: `DecompositionResult`

---

## `growth_metrics`

```python
def growth_metrics(
    sf: SalesFrame,
    window: int = 4,
) -> GrowthMetrics
```

**Behaviour**:
- Computes rolling period-over-period % change with the given `window`.
- Computes CAGR over the full series.
- Identifies inflection points (sign changes in rolling growth).

**Returns**: `GrowthMetrics`

---

## `detect_anomalies`

```python
def detect_anomalies(
    sf: SalesFrame,
    method: Literal["auto", "zscore", "iqr", "iforest", "contextual"] = "auto",
    threshold: Optional[float] = None,
) -> AnomalyTable
```

**Behaviour**:
- `"auto"`: selects `contextual` for daily/weekly univariate series; else `zscore`.
- `"contextual"`: compares each observation against peers in the same day-of-week or
  calendar-month bucket; uses z-score within bucket (default threshold=2.5).
- `"zscore"`: default threshold=3.0.
- `"iqr"`: default threshold=1.5 (IQR multiplier).
- `"iforest"`: sklearn IsolationForest, contamination=0.05; `threshold` ignored.
- Returns empty `AnomalyTable` (no rows) when no anomalies detected.

**Returns**: `AnomalyTable`

---

## `compare_segments`

```python
def compare_segments(
    sf: SalesFrame,
    by: Optional[str] = None,
    ranked_by: Literal["cagr", "latest_value"] = "cagr",
) -> SegmentRanking
```

**Behaviour**:
- Requires `sf.segment_col` to be set (or `by` to override it).
- Raises `ValueError` if no segment column is available.
- Computes per-segment CAGR, latest value, and rolling slope direction.
- Returns segments sorted by `ranked_by` descending.

**Returns**: `SegmentRanking`

---

## `plot_annotated`

```python
def plot_annotated(
    sf: SalesFrame,
    anomalies: Optional[AnomalyTable] = None,
    trend: Optional[DecompositionResult] = None,
    title: Optional[str] = None,
) -> plt.Figure
```

**Behaviour**:
- Plots the raw sales series.
- Overlays the trend line if `trend` is supplied.
- Marks anomaly dates in red with labels if `anomalies` is supplied.
- Returns the Figure object (does not call `plt.show()`; caller decides).

**Returns**: `matplotlib.figure.Figure`

---

## `narrate`

```python
def narrate(
    results: AnomalyTable | GrowthMetrics | SegmentRanking | DecompositionResult,
) -> None
```

**Behaviour**:
- Prints a plain-English summary of `results` to stdout.
- Format per result type:
  - `AnomalyTable`: lists each anomaly with date, direction, magnitude label.
  - `GrowthMetrics`: states trend direction, CAGR, and any inflection points.
  - `SegmentRanking`: names the top and bottom segments by CAGR.
  - `DecompositionResult`: states overall trend direction and seasonal pattern strength.
- If `results` contains no data (empty anomaly table, etc.), prints a "nothing notable" message.

**Returns**: `None` (always prints; never raises)
