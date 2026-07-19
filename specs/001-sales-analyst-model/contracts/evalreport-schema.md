# Contract: EvalReport JSON Schema

**Version**: 1.0.0 | **Date**: 2026-07-13

Defines the format for evaluation report files written to `eval/reports/`. One JSON object per
file; filename convention: `<model_variant>-<ISO-timestamp>.json`
(e.g. `1.5B-2026-07-13T14-30-00Z.json`).

---

## Schema

```json
{
  "model_variant":              "<'1.5B' | '3B'>",
  "model_name":                 "<Ollama model name string>",
  "salestools_version":         "<semver string>",
  "eval_set_size":              <integer>,
  "pass_at_1":                  <float 0.0–1.0>,
  "signal_detection_accuracy":  <float 0.0–1.0>,
  "scope_refusal_accuracy":     <float 0.0–1.0>,
  "timestamp":                  "<ISO-8601 UTC string>",
  "notes":                      "<free-text string>"
}
```

### Field Constraints

| Field                       | Type    | Constraint                                              |
|-----------------------------|---------|---------------------------------------------------------|
| model_variant               | string  | Exactly `"1.5B"` or `"3B"`                             |
| model_name                  | string  | Ollama model name, e.g. `"sales-analyst-1.5b"`          |
| salestools_version          | string  | Semver "MAJOR.MINOR.PATCH"                              |
| eval_set_size               | integer | ≥1; expected ~100 for held-out set                      |
| pass_at_1                   | float   | [0.0, 1.0] inclusive                                    |
| signal_detection_accuracy   | float   | [0.0, 1.0] inclusive                                    |
| scope_refusal_accuracy      | float   | [0.0, 1.0] inclusive                                    |
| timestamp                   | string  | ISO-8601 with UTC suffix Z                              |
| notes                       | string  | Free text; record hardware and any anomalies            |

---

## Metric Definitions

**pass@1**: Fraction of held-out code answers that execute without raising an exception in a
clean sandbox (same verifier used during training data generation). Scope-refusal pairs
(signal_type = `scope_refusal`) are excluded from this metric.

**signal_detection_accuracy**: Fraction of non-refusal held-out pairs where the executed code
correctly detects the planted signal (same programmatic check used during generation).

**scope_refusal_accuracy**: Fraction of `scope_refusal` held-out pairs where the generated
"code" contains no callable salestools function and contains a refusal indicator
(a comment string matching `/out.of.scope|outside.*scope|cannot|not.*support/i`).

---

## A/B Comparison Report

`eval/compare.py` reads two EvalReport files (one per model variant) and prints a side-by-side
table to stdout. No additional schema; output is human-readable text only.

---

## Versioning Policy

- Adding a new metric field: MINOR bump.
- Changing a metric definition: MAJOR bump (existing reports become incomparable).
- The `salestools_version` field allows cross-version comparisons (v1 eval vs. v2 eval).
