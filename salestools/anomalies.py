from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from salestools.core import AnomalyTable, SalesFrame, is_sub_monthly


def _empty_table(method: str, threshold: float) -> AnomalyTable:
    cols = ["date", "value", "score", "method", "label"]
    return AnomalyTable(anomalies=pd.DataFrame(columns=cols), method=method, threshold=threshold)


def _build_row(date, value, score, method, direction) -> dict:
    magnitude = abs(score)
    label = f"{magnitude:.1f}σ {'above' if direction > 0 else 'below'} expected"
    return {"date": date, "value": value, "score": score, "method": method, "label": label}


def _zscore_detect(series: pd.Series, threshold: float) -> pd.DataFrame:
    mean, std = series.mean(), series.std()
    if std == 0:
        return pd.DataFrame()
    z = (series - mean) / std
    mask = z.abs() > threshold
    rows = [
        _build_row(idx, float(series[idx]), float(z[idx]), "zscore", float(z[idx]))
        for idx in series[mask].index
    ]
    return pd.DataFrame(rows)


def _iqr_detect(series: pd.Series, threshold: float) -> pd.DataFrame:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    mask = (series < lower) | (series > upper)
    mean = series.mean()
    rows = [
        _build_row(idx, float(series[idx]), float((series[idx] - mean) / (iqr or 1)), "iqr",
                   1 if series[idx] > mean else -1)
        for idx in series[mask].index
    ]
    return pd.DataFrame(rows)


def _iforest_detect(series: pd.Series) -> pd.DataFrame:
    X = series.values.reshape(-1, 1)
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(X)
    scores = -clf.score_samples(X)
    mask = preds == -1
    mean = series.mean()
    rows = [
        _build_row(series.index[i], float(series.iloc[i]), float(scores[i]), "iforest",
                   1 if series.iloc[i] > mean else -1)
        for i in range(len(series)) if mask[i]
    ]
    return pd.DataFrame(rows)


def _contextual_detect(series: pd.Series, freq: str, threshold: float) -> pd.DataFrame:
    df = series.to_frame("value")
    if is_sub_monthly(freq):
        df["bucket"] = df.index.dayofweek
    else:
        df["bucket"] = df.index.month

    rows = []
    for bucket, group in df.groupby("bucket"):
        vals = group["value"]
        mean, std = vals.mean(), vals.std()
        if std == 0:
            continue
        z = (vals - mean) / std
        for idx in vals[z.abs() > threshold].index:
            zi = float(z[idx])
            rows.append(_build_row(idx, float(vals[idx]), zi, "contextual", zi))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def detect_anomalies(
    sf: SalesFrame,
    method: Literal["auto", "zscore", "iqr", "iforest", "contextual"] = "auto",
    threshold: Optional[float] = None,
) -> AnomalyTable:
    series = sf.data[sf.value_col].dropna()

    if method == "auto":
        method = "contextual" if is_sub_monthly(sf.freq) and sf.segment_col is None else "zscore"

    if method == "zscore":
        t = threshold if threshold is not None else 3.0
        df = _zscore_detect(series, t)
        return AnomalyTable(anomalies=df, method="zscore", threshold=t) if not df.empty else _empty_table("zscore", t)

    if method == "iqr":
        t = threshold if threshold is not None else 1.5
        df = _iqr_detect(series, t)
        return AnomalyTable(anomalies=df, method="iqr", threshold=t) if not df.empty else _empty_table("iqr", t)

    if method == "iforest":
        df = _iforest_detect(series)
        return AnomalyTable(anomalies=df, method="iforest", threshold=0.05) if not df.empty else _empty_table("iforest", 0.05)

    if method == "contextual":
        t = threshold if threshold is not None else 2.5
        df = _contextual_detect(series, sf.freq, t)
        return AnomalyTable(anomalies=df, method="contextual", threshold=t) if not df.empty else _empty_table("contextual", t)

    raise ValueError(f"Unknown method: {method!r}")
