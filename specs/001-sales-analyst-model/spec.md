# Feature Specification: Local Sales-Analyst Codegen Model

**Feature Branch**: `001-sales-analyst-model`

**Created**: 2026-07-13

**Status**: Draft

**Input**: A locally-run fine-tuned small language model that turns natural-language questions about
sales data into runnable `salestools` Python code inside Jupyter, specialized in trend analysis
and anomaly detection.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask a Trend Question (Priority: P1)

A small-business owner or analyst opens a Jupyter notebook, loads their sales CSV, and types a
natural-language question using the `%%ask` cell magic. The local model generates a short
`salestools` script into the next cell. Running it produces an annotated chart and a printed
plain-English summary of findings.

**Why this priority**: This is the core end-user interaction — without it no value is delivered.

**Independent Test**: Load a sample sales CSV, run `%%ask "is my overall trend up or down?"`,
verify the generated code executes without edits and prints a trend summary with a decomposition
chart.

**Acceptance Scenarios**:

1. **Given** a sales CSV with date and amount columns, **When** the user asks a trend question
   via `%%ask`, **Then** generated code runs without edits and prints a trend summary with a
   decomposition chart.
2. **Given** no internet connection, **When** the user runs `%%ask`, **Then** inference still
   works (local model, no network calls).
3. **Given** a question outside scope (e.g., "write me a web scraper"), **Then** the model
   responds with a statement that the question is outside sales analysis scope.

---

### User Story 2 - Anomaly Detection Question (Priority: P2)

The analyst asks "any unusual days?" and the generated code identifies and annotates anomalies on
a chart, with a plain-English explanation for each flagged point.

**Why this priority**: Anomaly detection is the second most common use-case and is explicitly
part of the demo headline.

**Independent Test**: Use a dataset with a planted anomaly; run `%%ask "any unusual days?"`,
verify the generated code flags the correct anomaly on an annotated chart.

**Acceptance Scenarios**:

1. **Given** a dataset containing a planted anomaly, **When** the user asks "any unusual days?",
   **Then** generated code flags the planted anomaly on an annotated chart with a plain-English
   explanation.
2. **Given** a dataset where a low value occurs consistently on the same weekday (not an anomaly),
   **Then** the generated code does NOT flag it as anomalous.

---

### User Story 3 - Multi-Product Comparison (Priority: P3)

The analyst asks "which product is dragging sales down?" and the generated code produces a
per-product trend comparison that correctly identifies the underperformer.

**Why this priority**: Segment comparison broadens the demo and demonstrates the model handles
multi-entity questions.

**Independent Test**: Use a multi-product dataset with one planted underperforming product; run
`%%ask "which product is dragging sales down?"`, verify generated code identifies the correct
product.

**Acceptance Scenarios**:

1. **Given** a multi-product dataset with one underperforming product, **When** the user asks
   "which product is dragging sales down?", **Then** generated code produces a per-product trend
   comparison and correctly identifies the underperformer.

---

### User Story 4 - Model Size A/B Evaluation (Priority: P2)

Both a 1.5B-parameter and a 3B-parameter model variant are fine-tuned and evaluated on the
held-out set side-by-side, producing a report that quantifies the quality-vs-speed tradeoff
between the two sizes.

**Why this priority**: The A/B comparison is an explicit project requirement and directly informs
which model variant to recommend for production use.

**Independent Test**: Run the evaluation harness against both fine-tuned checkpoints; verify a
report is produced comparing pass@1, signal-detection accuracy, and scope-refusal accuracy for
each variant.

**Acceptance Scenarios**:

1. **Given** both the 1.5B and 3B model variants are fine-tuned and available locally, **When**
   the evaluation harness runs on the held-out set, **Then** a report is produced comparing
   pass@1, signal-detection accuracy, and scope-refusal accuracy for each variant.
2. **Given** the 1.5B model, **When** used via `%%ask` on a CPU-only laptop, **Then** inference
   completes in usable time (well under 1 minute per query).

---

### User Story 5 - Library Evolution Demo (Priority: P3, Phase 2)

The `salestools` library gains two new functions in v2. The data generator reruns on only the new
API surface, an incremental fine-tune is performed, and a before/after evaluation confirms the
model adopts v2 functions while retaining v1 skills.

**Why this priority**: Validates the versioned-library lifecycle required by the project
constitution; deferred to Phase 2, specified now so that v1 design does not foreclose it.

**Independent Test**: After the v2 incremental fine-tune, run the evaluation harness comparing
the v1-only checkpoint against the v2 checkpoint on v2-specific questions; verify the v2 model
uses the new functions and the v1 model does not.

**Acceptance Scenarios**:

1. **Given** the v2 library with `forecast` and `cohort_analysis` functions added, **When** an
   incremental fine-tune is performed on the delta dataset, **Then** the resulting model generates
   code using v2 functions for relevant questions.
2. **Given** the v2 fine-tuned model, **When** asked v1 questions, **Then** it still uses v1
   functions correctly (no regression).

---

### Edge Cases

- CSV with missing dates or gaps → the library fills or flags gaps explicitly rather than
  silently producing incorrect results.
- Dataset too short for seasonal decomposition (fewer than 2 seasonal cycles) → functions degrade
  gracefully with a clear printed message rather than an error.
- Ambiguous question → model defaults to the most common interpretation and states this assumption
  in the printed summary.
- Local model runtime not available → `%%ask` shows a friendly setup hint, not a stack trace.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a `salestools` pip-installable library with functions for
  loading, trend-decomposing, computing growth metrics, detecting anomalies, comparing segments,
  annotating charts, and narrating findings from sales time-series data.
- **FR-002**: The `salestools` library MUST validate and regularize input sales data, explicitly
  handling missing dates and gaps rather than silently accepting malformed input.
- **FR-003**: The system MUST generate a verified synthetic training dataset of 800–1,500
  question-to-`salestools`-code pairs; each pair MUST be confirmed by execution to both run
  without error and detect the planted signal in the dataset.
- **FR-004**: The data generator MUST be versioned and rerunnable against any `salestools` release
  so that future library updates can produce incremental delta datasets without regenerating the
  full corpus.
- **FR-005**: Two model variants (1.5B-parameter and 3B-parameter) MUST each be fine-tuned and
  exported as locally-runnable artifacts.
- **FR-006**: Both model variants MUST be evaluated on a held-out set (never used in training)
  and compared on pass@1 (code executes without error), signal-detection accuracy (planted signal
  found), and scope-refusal accuracy (out-of-scope questions correctly declined).
- **FR-007**: The system MUST provide a `%%ask` Jupyter cell magic that sends a natural-language
  question to the local model and inserts the generated `salestools` code into the next cell
  without auto-executing it.
- **FR-008**: Generated code MUST be ≤15 lines and MUST end with a printed plain-English summary
  of findings.
- **FR-009**: The `%%ask` magic MUST work fully offline; no network calls are permitted at
  inference time.
- **FR-010**: The `%%ask` magic MUST surface a user-friendly setup hint when the local model
  runtime is not available, rather than exposing a stack trace.
- **FR-011**: The v1 library and data-generator design MUST not foreclose the v2 lifecycle: new
  functions can be added and an incremental fine-tune performed without modifying the generator's
  core architecture.
- **FR-012**: Training MUST be fully reproducible via seeded randomness, pinned dependencies, and
  versioned configuration files checked into the repository.

### Key Entities

- **SalesFrame**: The core data structure wrapping a validated, regularized sales time series;
  carries metadata (date column, value column, detected frequency, any gap flags).
- **TrainingPair**: A verified (question, `salestools`-code) pair with metadata including planted
  signal type, execution verification status, and the `salestools` version it was generated
  against.
- **EvalReport**: A structured comparison of model performance metrics (pass@1, signal-detection
  accuracy, scope-refusal accuracy) across model variants (1.5B vs. 3B) and library versions.
- **ModelArtifact**: A fine-tuned, locally-servable model checkpoint associated with a specific
  `salestools` library version and parameter size.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The fine-tuned 1.5B model achieves pass@1 ≥ 85% on the held-out evaluation set
  (the base model is expected to score well below this threshold).
- **SC-002**: Signal-detection accuracy ≥ 80% on held-out planted-anomaly questions for the 1.5B
  model.
- **SC-003**: The A/B evaluation report quantifies the quality gap between the 1.5B and 3B
  variants, enabling an informed model selection decision without additional manual testing.
- **SC-004**: A new user can go from a fresh machine (with the local model runtime installed) to
  answering their first sales question in under 10 minutes of setup.
- **SC-005**: The 1.5B model responds to a `%%ask` query on a CPU-only laptop in well under
  1 minute per query.
- **SC-006**: 100% of training pairs included in the dataset have been verified by execution
  against their planted signal; no unverified pairs enter training.
- **SC-007**: The v2 lifecycle demo shows measurable improvement on v2-function questions after
  incremental fine-tune, with no regression on v1-question accuracy.

## Assumptions

- Training is performed on Google Colab Pro (L4 or A100 GPU); a local GPU is not required but
  may be used as a convenience.
- The 3B model variant requires Colab Pro (rather than free-tier hardware) to complete fine-tuning
  within practical time limits.
- English-only natural-language questions for v1; multi-language support is out of scope.
- Input data is a single flat sales table (date, amount, optional product/region columns);
  SQL databases and multi-table schemas are out of scope for v1.
- The `%%ask` magic does not auto-execute generated code; the analyst reviews and runs it
  manually.
- Out-of-scope questions are handled by the model generating a plain-English refusal comment
  in the code cell, not by a runtime error or silent failure.

## Out of Scope (v1)

Real customer data, SQL databases, forecasting (deferred to v2 lifecycle demo), multi-turn
conversation, auto-executing generated code, web UI, non-English questions.
