"""Question template library.

Maps each signal_type to a list of natural-language question paraphrases.
The code template (expected salestools answer) lives alongside each entry.
"""
from __future__ import annotations

TEMPLATES: dict[str, list[dict]] = {
    "trend_up": [
        {"q": "Is my overall sales trend going up?",
         "code": "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"},
        {"q": "Has my revenue been increasing over time?",
         "code": "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"},
        {"q": "What direction is my sales trend heading?",
         "code": "sf = load_sales('data.csv')\ngm = growth_metrics(sf)\nnarrate(gm)"},
        {"q": "Show me whether sales are growing.",
         "code": "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nfig = plot_annotated(sf, trend=result)\nnarrate(result)"},
        {"q": "Is there a positive trend in my data?",
         "code": "sf = load_sales('data.csv')\ngm = growth_metrics(sf)\nnarrate(gm)"},
    ],
    "trend_down": [
        {"q": "Is my overall sales trend going down?",
         "code": "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"},
        {"q": "Has revenue been declining?",
         "code": "sf = load_sales('data.csv')\ngm = growth_metrics(sf)\nnarrate(gm)"},
        {"q": "Are sales falling over time?",
         "code": "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"},
        {"q": "What's the sales trajectory — is it negative?",
         "code": "sf = load_sales('data.csv')\ngm = growth_metrics(sf)\nnarrate(gm)"},
        {"q": "Show me if there's a downward trend.",
         "code": "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nfig = plot_annotated(sf, trend=result)\nnarrate(result)"},
    ],
    "anomaly_spike": [
        {"q": "Which weeks had unusually high sales?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Were there any sales spikes in my data?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Show me any periods of abnormally high revenue.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='zscore')\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Identify weeks where sales were far above normal.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nnarrate(anomalies)"},
        {"q": "Were there any surprising jumps in my sales figures?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
    ],
    "anomaly_drop": [
        {"q": "Which weeks had unusually low sales?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Were there any sales dips I should know about?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Show me periods where revenue dropped unexpectedly.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='zscore')\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Highlight any weeks where sales were far below average.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nnarrate(anomalies)"},
        {"q": "Flag any unusual downturns in my sales data.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
    ],
    "anomaly_contextual": [
        {"q": "Are there any days that are unusual for their day of week?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='contextual')\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Find anomalies that are odd compared to similar days.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='contextual')\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Which days stand out compared to the same weekday historically?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='contextual')\nnarrate(anomalies)"},
        {"q": "Show me contextual anomalies in my daily sales.",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='contextual')\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
        {"q": "Are there any Mondays (or other weekdays) that performed unexpectedly?",
         "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf, method='contextual')\nfig = plot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)"},
    ],
    "segment_drag": [
        {"q": "Which product is dragging my overall sales down?",
         "code": "sf = load_sales('data.csv', segment_col='product')\nranking = compare_segments(sf)\nnarrate(ranking)"},
        {"q": "Which segment has the worst growth trend?",
         "code": "sf = load_sales('data.csv', segment_col='product')\nranking = compare_segments(sf)\nnarrate(ranking)"},
        {"q": "Compare my products by sales growth and identify the laggard.",
         "code": "sf = load_sales('data.csv', segment_col='product')\nranking = compare_segments(sf)\nnarrate(ranking)"},
        {"q": "Show me which product or region is underperforming.",
         "code": "sf = load_sales('data.csv', segment_col='product')\nranking = compare_segments(sf)\nnarrate(ranking)"},
        {"q": "Rank my segments from best to worst growth.",
         "code": "sf = load_sales('data.csv', segment_col='product')\nranking = compare_segments(sf)\nnarrate(ranking)"},
    ],
    "scope_refusal": [
        {"q": "Write me a web scraper for competitor prices.",
         "code": "# This question is outside the scope of salestools sales analysis."},
        {"q": "Can you help me write a SQL query?",
         "code": "# This question is outside the scope of salestools sales analysis."},
        {"q": "Predict tomorrow's stock price.",
         "code": "# This question is outside the scope of salestools sales analysis."},
        {"q": "Build a recommendation engine for my products.",
         "code": "# This question is outside the scope of salestools sales analysis."},
        {"q": "What's the weather forecast for next week?",
         "code": "# This question is outside the scope of salestools sales analysis."},
    ],
    # v2 signal types
    "forecast_up": [
        {"q": "What will my sales look like over the next 12 months?",
         "code": "sf = load_sales('data.csv')\nresult = forecast(sf, horizon=12)\nnarrate(result)"},
        {"q": "Can you forecast my revenue for the next quarter?",
         "code": "sf = load_sales('data.csv')\nresult = forecast(sf, horizon=3)\nnarrate(result)"},
        {"q": "Project my sales trend forward for the next 6 periods.",
         "code": "sf = load_sales('data.csv')\nresult = forecast(sf, horizon=6)\nnarrate(result)"},
        {"q": "Show me a sales forecast with confidence intervals.",
         "code": "sf = load_sales('data.csv')\nresult = forecast(sf, horizon=12)\nfig = result.fig\nnarrate(result)"},
        {"q": "What does the model predict for my future sales?",
         "code": "sf = load_sales('data.csv')\nresult = forecast(sf, horizon=12)\nnarrate(result)"},
    ],
    "cohort_question": [
        {"q": "How is customer retention trending across cohorts?",
         "code": "sf = load_sales('data.csv', segment_col='cohort')\ntable = cohort_analysis(sf, cohort_col='cohort')\nnarrate(table)"},
        {"q": "Show me a cohort retention analysis.",
         "code": "sf = load_sales('data.csv', segment_col='cohort')\ntable = cohort_analysis(sf, cohort_col='cohort')\nnarrate(table)"},
        {"q": "Which customer cohort has the best retention?",
         "code": "sf = load_sales('data.csv', segment_col='cohort')\ntable = cohort_analysis(sf, cohort_col='cohort')\nnarrate(table)"},
        {"q": "Are later cohorts retaining customers better than earlier ones?",
         "code": "sf = load_sales('data.csv', segment_col='cohort')\ntable = cohort_analysis(sf, cohort_col='cohort')\nnarrate(table)"},
        {"q": "Build a cohort analysis heatmap for my sales data.",
         "code": "sf = load_sales('data.csv', segment_col='cohort')\ntable = cohort_analysis(sf, cohort_col='cohort')\nfig = table.fig\nnarrate(table)"},
    ],
}


def get_questions(signal_type: str) -> list[dict]:
    """Return list of {q, code} dicts for the given signal_type."""
    if signal_type not in TEMPLATES:
        raise ValueError(f"Unknown signal_type: {signal_type!r}. Valid: {list(TEMPLATES)}")
    return TEMPLATES[signal_type]
