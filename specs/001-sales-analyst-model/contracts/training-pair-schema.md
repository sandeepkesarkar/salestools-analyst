# Contract: Training Pair JSONL Schema

**Version**: 1.0.0 | **Date**: 2026-07-13

Defines the wire format for training pairs written to `data/v1/train.jsonl` and
`data/v1/held_out.jsonl`. One JSON object per line; UTF-8 encoding; no trailing commas.

---

## Schema

```json
{
  "id":                  "<uuid4 string>",
  "question":            "<natural-language question string>",
  "code":                "<salestools Python code string; ≤15 lines>",
  "signal_type":         "<see Signal Types>",
  "verified":            true,
  "salestools_version":  "<semver string>",
  "dataset_seed":        <integer>,
  "paraphrase_id":       <integer>
}
```

### Field Constraints

| Field               | Type    | Required | Constraint                                                  |
|---------------------|---------|----------|-------------------------------------------------------------|
| id                  | string  | yes      | UUID v4 format                                              |
| question            | string  | yes      | Non-empty; English; ends with "?"                           |
| code                | string  | yes      | Non-empty; ≤15 newline-separated lines; valid Python        |
| signal_type         | string  | yes      | One of the Signal Types listed below                        |
| verified            | boolean | yes      | MUST be `true`; unverified pairs are never written to file  |
| salestools_version  | string  | yes      | Semver format "MAJOR.MINOR.PATCH"                           |
| dataset_seed        | integer | yes      | ≥0; seeds 9000–9999 reserved for held-out set              |
| paraphrase_id       | integer | yes      | ≥0; index into question template's paraphrase list          |

### Signal Types

| Value               | Description                                                    |
|---------------------|----------------------------------------------------------------|
| `trend_up`          | Dataset contains a planted upward linear trend                 |
| `trend_down`        | Dataset contains a planted downward linear trend               |
| `anomaly_spike`     | Dataset contains a planted above-baseline spike                |
| `anomaly_drop`      | Dataset contains a planted below-baseline drop                 |
| `anomaly_contextual`| Dataset contains a contextual anomaly (unusual for its bucket) |
| `segment_drag`      | One segment has a planted downward trend in a multi-segment CSV|
| `scope_refusal`     | Question is outside salestools scope; code MUST be a refusal comment |

---

## Chat Format (for fine-tuning)

Training pairs are converted to ChatML format for Unsloth/PEFT consumption:

```json
{
  "messages": [
    {"role": "system",    "content": "<fixed system prompt>"},
    {"role": "user",      "content": "<question>"},
    {"role": "assistant", "content": "<code>"}
  ]
}
```

The system prompt is fixed and stored at `training/config/system_prompt.txt`.

---

## Versioning Policy

- Adding a new `signal_type` value: MINOR bump (additive).
- Adding a new required field: MAJOR bump (breaks existing readers).
- Changing a field type or constraint: MAJOR bump.
- The `salestools_version` field ensures older pairs can be identified and excluded when
  evaluating against a newer library version.
