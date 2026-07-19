# Data Model: Local Sales-Analyst Codegen Model

**Phase**: 1 | **Date**: 2026-07-13 | **Plan**: [plan.md](plan.md)

---

## SalesFrame

The central data structure. Wraps a validated, frequency-regularized sales time series.

```python
@dataclass
class SalesFrame:
    data: pd.DataFrame          # DatetimeIndex, single value column (value_col)
    date_col: str               # original column name from source CSV
    value_col: str              # original column name for the sales metric
    freq: str                   # detected/user-supplied: 'D' | 'W' | 'M' | 'Q' | 'Y'
    segment_col: Optional[str]  # product/region column, None for univariate series
    gap_flags: pd.Series        # boolean mask aligned with data.index; True = gap filled
```

**Validation rules**:
- `date_col` must parse to `datetime64`; raises `ValueError` on failure.
- `value_col` must be numeric; raises `TypeError` on non-numeric.
- Gaps ≤ 3 consecutive periods: forward-fill and set `gap_flags=True`.
- Gaps > 3 consecutive periods: flag as gap (no fill); downstream functions warn explicitly.
- Minimum length: 4 periods (hard minimum for any analysis); raises `ValueError` if shorter.
- Minimum for seasonal decomposition: 2 full seasonal cycles; `decompose_trend` degrades
  gracefully with a printed warning if this threshold is not met.

**State transitions**:
```
raw CSV
  → load_sales() → SalesFrame (validated, regularized)
    → decompose_trend()  → DecompositionResult
    → detect_anomalies() → AnomalyTable
    → compare_segments() → SegmentRanking
    → growth_metrics()   → GrowthMetrics
    → plot_annotated()   → matplotlib Figure (side effect)
    → narrate()          → None (prints to stdout)
```

---

## DecompositionResult

Output of `decompose_trend()`.

```python
@dataclass
class DecompositionResult:
    trend: pd.Series        # STL trend component
    seasonal: pd.Series     # STL seasonal component
    residual: pd.Series     # STL residual component
    period: int             # seasonal period used (auto-detected or user-supplied)
    fig: Optional[plt.Figure]  # decomposition chart (returned for inspection/saving)
```

---

## AnomalyTable

Output of `detect_anomalies()`.

```python
@dataclass
class AnomalyTable:
    anomalies: pd.DataFrame  # columns: date, value, score, method, label
    method: str              # 'zscore' | 'iqr' | 'iforest' | 'contextual'
    threshold: float         # detection threshold used
```

**`anomalies` DataFrame schema**:

| column  | dtype    | description                                          |
|---------|----------|------------------------------------------------------|
| date    | datetime | timestamp of the anomalous observation               |
| value   | float    | observed sales value                                 |
| score   | float    | anomaly score (z-score, IQR multiple, or IF score)   |
| method  | str      | detection method applied to this row                 |
| label   | str      | human-readable label, e.g. "3.1σ above expected"     |

**Detection methods**:
- `zscore`: |z| > 3.0 (configurable)
- `iqr`: value outside Q1 − 1.5×IQR or Q3 + 1.5×IQR
- `iforest`: sklearn IsolationForest, contamination=0.05
- `contextual`: compares within same day-of-week / calendar-month bucket; uses z-score within bucket
- `auto`: selects `contextual` if `segment_col` is None and series is daily/weekly; else `zscore`

---

## GrowthMetrics

Output of `growth_metrics()`.

```python
@dataclass
class GrowthMetrics:
    rolling_growth: pd.Series    # period-over-period % change, rolling window
    cagr: float                  # compound annual growth rate over full series
    inflection_points: pd.Series # dates where rolling growth changes sign
    window: int                  # rolling window used
```

---

## SegmentRanking

Output of `compare_segments()`.

```python
@dataclass
class SegmentRanking:
    summary: pd.DataFrame    # columns: segment, cagr, latest_value, trend_direction
    ranked_by: str           # 'cagr' | 'latest_value' (default: 'cagr')
```

**`summary` DataFrame schema**:

| column          | dtype  | description                                     |
|-----------------|--------|-------------------------------------------------|
| segment         | str    | product / region value                          |
| cagr            | float  | compound annual growth rate for this segment    |
| latest_value    | float  | most recent period sales value                  |
| trend_direction | str    | 'up' | 'down' | 'flat' (based on rolling slope)  |

---

## TrainingPair

A single verified (question → code) example in the training dataset.

```python
@dataclass
class TrainingPair:
    id: str                    # uuid4
    question: str              # natural-language question (user turn)
    code: str                  # salestools code answer (assistant turn); ≤15 lines
    signal_type: str           # see Signal Types below
    verified: bool             # True only after sandbox execution + signal check pass
    salestools_version: str    # semver string, e.g. '1.0.0'
    dataset_seed: int          # RNG seed used to generate the verification dataset
    paraphrase_id: int         # index into the paraphrase table for this template
```

**Signal types**:
- `trend_up` — dataset has a planted upward trend
- `trend_down` — dataset has a planted downward trend
- `anomaly_spike` — dataset has a planted above-baseline spike on a specific date
- `anomaly_drop` — dataset has a planted below-baseline drop on a specific date
- `anomaly_contextual` — dataset has a contextual anomaly (unusual for its weekday/month bucket)
- `segment_drag` — one segment has a planted downward trend; others are flat/up
- `scope_refusal` — question is outside salestools scope; correct answer is a refusal comment

**JSONL serialization** (one object per line):
```json
{
  "id": "550e8400-...",
  "question": "Which weeks had unusual sales this year?",
  "code": "sf = load_sales('data.csv')\nanomalies = detect_anomalies(sf)\nplot_annotated(sf, anomalies=anomalies)\nnarrate(anomalies)",
  "signal_type": "anomaly_spike",
  "verified": true,
  "salestools_version": "1.0.0",
  "dataset_seed": 42,
  "paraphrase_id": 2
}
```

---

## EvalReport

Output of the evaluation harness for a single model variant run.

```python
@dataclass
class EvalReport:
    model_variant: str                # '1.5B' | '3B'
    model_name: str                   # Ollama model name, e.g. 'sales-analyst-1.5b'
    salestools_version: str           # library version evaluated against
    eval_set_size: int                # number of held-out pairs evaluated
    pass_at_1: float                  # [0.0–1.0] fraction of codes that execute without error
    signal_detection_accuracy: float  # [0.0–1.0] fraction that detect planted signal
    scope_refusal_accuracy: float     # [0.0–1.0] fraction of OOS questions correctly declined
    timestamp: str                    # ISO-8601, e.g. '2026-07-13T14:30:00Z'
    notes: str                        # free-text, e.g. hardware used
```

**JSON serialization** (eval/reports/<model_variant>-<timestamp>.json):
```json
{
  "model_variant": "1.5B",
  "model_name": "sales-analyst-1.5b",
  "salestools_version": "1.0.0",
  "eval_set_size": 100,
  "pass_at_1": 0.88,
  "signal_detection_accuracy": 0.83,
  "scope_refusal_accuracy": 0.95,
  "timestamp": "2026-07-13T14:30:00Z",
  "notes": "Colab Pro L4, q4_k_m"
}
```

---

## ModelArtifact

Metadata record for a fine-tuned model checkpoint (stored alongside the GGUF file).

```python
@dataclass
class ModelArtifact:
    model_name: str          # Ollama model name, e.g. 'sales-analyst-1.5b'
    parameter_size: str      # '1.5B' | '3B'
    base_model: str          # HuggingFace model id, e.g. 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
    salestools_version: str  # library version trained against
    quantization: str        # 'q4_k_m'
    lora_rank: int           # 16
    lora_alpha: int          # 32
    training_seed: int       # global seed used for all RNG in training
    colab_hardware: str      # 'L4' | 'A100'
    training_minutes: int    # wall-clock training time
    gguf_path: str           # local path, e.g. 'models/gguf/sales-analyst-1.5b.q4_k_m.gguf'
```

**JSON serialization** (models/gguf/<name>.meta.json):
```json
{
  "model_name": "sales-analyst-1.5b",
  "parameter_size": "1.5B",
  "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "salestools_version": "1.0.0",
  "quantization": "q4_k_m",
  "lora_rank": 16,
  "lora_alpha": 32,
  "training_seed": 42,
  "colab_hardware": "L4",
  "training_minutes": 45,
  "gguf_path": "models/gguf/sales-analyst-1.5b.q4_k_m.gguf"
}
```
