"""Sandboxed pair verifier.

Runs generated salestools code in an isolated subprocess with a 30-second timeout.
Uses a top-level worker function (required for multiprocessing pickling on macOS/Python 3.13).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import traceback

import pandas as pd


def _run_code_worker(code: str, csv_path: str, signal_type: str, queue: mp.Queue) -> None:
    """Top-level worker — picklable on all platforms."""
    try:
        import matplotlib
        matplotlib.use("Agg")

        from salestools import (
            compare_segments, decompose_trend, detect_anomalies,
            growth_metrics, load_sales, narrate, plot_annotated,
        )

        # Replace the placeholder path in code with the actual temp CSV path
        exec_code = code.replace("'data.csv'", repr(csv_path)).replace('"data.csv"', repr(csv_path))

        namespace: dict = {
            "load_sales": load_sales,
            "decompose_trend": decompose_trend,
            "growth_metrics": growth_metrics,
            "detect_anomalies": detect_anomalies,
            "compare_segments": compare_segments,
            "plot_annotated": plot_annotated,
            "narrate": narrate,
        }
        exec(exec_code, namespace)  # noqa: S102

        detected = _detect(namespace, signal_type)
        queue.put((True, detected, ""))
    except Exception:
        queue.put((False, False, traceback.format_exc()))


def _detect(namespace: dict, signal_type: str) -> bool:
    """Signal-type-specific detection — runs inside the subprocess."""
    if signal_type == "scope_refusal":
        return True

    for obj in namespace.values():
        if signal_type == "trend_up":
            if hasattr(obj, "cagr") and isinstance(obj.cagr, float) and obj.cagr > 0.02:
                return True
            if hasattr(obj, "trend") and hasattr(obj.trend, "iloc") and len(obj.trend.dropna()) > 1:
                t = obj.trend.dropna()
                if t.iloc[-1] > t.iloc[0]:
                    return True

        elif signal_type == "trend_down":
            if hasattr(obj, "cagr") and isinstance(obj.cagr, float) and obj.cagr < -0.02:
                return True
            if hasattr(obj, "trend") and hasattr(obj.trend, "iloc") and len(obj.trend.dropna()) > 1:
                t = obj.trend.dropna()
                if t.iloc[-1] < t.iloc[0]:
                    return True

        elif signal_type in ("anomaly_spike", "anomaly_drop", "anomaly_contextual"):
            if hasattr(obj, "anomalies") and hasattr(obj.anomalies, "empty"):
                if not obj.anomalies.empty:
                    return True

        elif signal_type == "segment_drag":
            if hasattr(obj, "summary") and hasattr(obj.summary, "iloc") and not obj.summary.empty:
                bottom = obj.summary.iloc[-1]
                seg = str(bottom.get("segment", "")).upper()
                if seg == "C":
                    return True

    return False


def verify_pair(
    code: str,
    dataset: pd.DataFrame,
    _detection_fn,  # kept for API compat but detection is now done inline by signal_type
    timeout: int = 30,
    signal_type: str = "",
) -> tuple[bool, str]:
    """Verify a (code, dataset) pair. Returns (passed, error_message)."""
    stripped = code.strip()
    if stripped.startswith("#") and ("outside" in stripped.lower() or "scope" in stripped.lower()):
        return True, ""

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        dataset.to_csv(f, index=False)
        csv_path = f.name

    try:
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_run_code_worker,
            args=(code, csv_path, signal_type, queue),
            daemon=True,
        )
        proc.start()
        proc.join(timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            return False, f"Timeout after {timeout}s"

        if queue.empty():
            return False, "Worker produced no result (likely crashed on import)"

        success, detected, error = queue.get_nowait()
        if not success:
            return False, error
        if not detected:
            return False, "Code ran cleanly but planted signal was not detected"
        return True, ""
    finally:
        os.unlink(csv_path)
