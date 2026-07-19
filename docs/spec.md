# Feature Specification: Local Sales-Analyst Codegen Model

<!-- Paste this content (or a summary of it) when running: /speckit.specify -->

**Feature Branch**: `001-sales-analyst-model`
**Status**: Draft
**Input**: A locally-run fine-tuned small language model that turns natural-language questions about sales data into runnable `salestools` Python code inside Jupyter, specialized in trend analysis and anomaly detection.

---

## User Scenarios

### Primary User Story
A small-business owner (or analyst) opens a Jupyter notebook, loads their sales CSV, and types:

```
%%ask
Which weeks this year had unusual sales, and is my overall trend up or down?
```

The local model (via Ollama) generates a short `salestools` script into the next cell. Running it produces an annotated chart (anomalies marked in red with explanations) and a printed plain-English summary, e.g. "Week 23 was 3.1σ above expected — likely promo spike. Underlying trend: +4% monthly growth after removing seasonality."

### Acceptance Scenarios
1. **Given** a sales CSV with date/amount columns, **When** the user asks a trend question via `%%ask`, **Then** generated code runs without edits and prints a trend summary with a decomposition chart.
2. **Given** a dataset containing a planted anomaly, **When** the user asks "any unusual days?", **Then** the generated code flags the planted anomaly on an annotated chart.
3. **Given** a multi-product dataset, **When** the user asks "which product is dragging sales down?", **Then** generated code produces a per-product trend comparison identifying the correct product.
4. **Given** no internet connection, **When** the user runs `%%ask`, **Then** everything still works (local Ollama inference).
5. **Given** a question outside scope (e.g., "write me a web scraper"), **Then** the model responds with a code comment stating the question is outside `salestools` scope.

### Edge Cases
- CSV with missing dates / gaps → `salestools.load_sales` fills or flags gaps explicitly.
- Dataset too short for seasonal decomposition (<2 seasonal cycles) → functions degrade gracefully with a clear message.
- Ambiguous question → model defaults to the most common interpretation and says so in the printed summary.
- Ollama not running → `%%ask` shows a friendly setup hint, not a stack trace.

---

## Requirements

### FR-1: `salestools` library (v1)
A pip-installable package with a deliberately small surface (~8 functions):
- `load_sales(path, date_col=..., value_col=...) -> SalesFrame` — parse, validate, regularize frequency
- `decompose_trend(sf, period='auto')` — STL trend/seasonal/residual + plot
- `growth_metrics(sf, window=...)` — rolling growth, inflection points
- `detect_anomalies(sf, method='auto'|'zscore'|'iqr'|'iforest'|'contextual')` — returns anomaly table
- `contextual` method compares like-with-like (day-of-week/month effects)
- `compare_segments(sf, by='product'|'region')` — per-segment trend ranking
- `plot_annotated(sf, anomalies=..., trend=...)` — chart with red markers + labels
- `narrate(results)` — plain-English findings summary (rule-based, prints)

### FR-2: Synthetic dataset generator
- Generates fake sales datasets with **planted, machine-checkable** signals: upward/downward trends, seasonality, promo spikes, drop-out days, contextual anomalies, segment divergence.
- Generates (NL question → salestools code) pairs across question templates + paraphrases.
- Executes every pair in a sandboxed kernel; keeps only pairs that (a) run clean and (b) detect the planted signal.
- Target: 800–1,500 verified pairs; outputs JSONL in chat format; versioned per salestools release.

### FR-3: Fine-tuning pipeline
- QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct via Unsloth; single notebook/script runnable on Colab T4.
- Merge adapter → convert to GGUF (q4_k_m) → `Modelfile` → `ollama create sales-analyst`.

### FR-4: Jupyter integration
- IPython extension providing `%%ask` cell magic: sends question + a fixed system prompt to local Ollama, inserts returned code into a new cell (not auto-executed).
- Config for model name and temperature.

### FR-5: Evaluation harness
- Held-out set of ~100 verified pairs never used in training.
- Metrics: pass@1 (execution success), signal-detection accuracy, scope-refusal accuracy.
- Report comparing base model vs. fine-tuned model (the headline demo number).

### FR-6 (Phase 2, spec'd not built): Library evolution demo
- `salestools` v2 adds `forecast(sf, horizon=...)` and `cohort_analysis(sf, ...)`.
- Data generator reruns on the API delta only; incremental fine-tune; before/after eval shows the model using new v2 functions correctly while retaining v1 skills.

### Non-Functional Requirements
- Inference: usable latency on CPU-only laptop via Ollama (1.5B q4).
- Generated code: ≤15 lines, always ends with `narrate(...)` or a printed summary.
- Fully offline at inference; reproducible training (seeded, pinned deps).

---

## Key Assumptions
- Training environment: free-tier Google Colab T4 (no local GPU assumed). [Flag if you have a local GPU — changes nothing structurally, just convenience.]
- English-only questions for v1.
- Single-table sales data (date, amount, optional product/region columns); no databases in v1.

## Success Criteria (measurable)
- Fine-tuned model pass@1 ≥ 85% on held-out set (base model expected well below).
- Signal-detection accuracy ≥ 80% on planted-anomaly questions.
- End-to-end demo: fresh machine → `pip install` + `ollama pull` + notebook → answered question in <10 minutes of setup.

## Out of Scope (v1)
- Real customer data, databases/SQL, forecasting (v2), multi-turn conversation, auto-executing generated code, web UI.
