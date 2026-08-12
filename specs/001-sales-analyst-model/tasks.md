---
description: "Task list for Local Sales-Analyst Codegen Model"
---

# Tasks: Local Sales-Analyst Codegen Model

**Input**: Design documents from `specs/001-sales-analyst-model/`

**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | contracts/ ✅

**Tests**: Unit + integration tests included in Polish phase only (not requested as TDD).

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks in same phase)
- **[Story]**: Which user story this task belongs to (US1–US5)
- All paths are repo-root-relative

---

## Phase 1: Setup (Project Initialization)

**Purpose**: Create directory scaffold and dependency manifests.

- [X] T001 Create all project directories: `salestools/`, `data/generator/`, `data/v1/`, `data/seeds/`, `training/config/`, `models/gguf/`, `models/adapters/`, `jupyter_magic/`, `eval/reports/`, `tests/unit/`, `tests/integration/`, `tests/fixtures/`
- [X] T002 Create `pyproject.toml`: package name `salestools`, version `1.0.0`, deps `pandas>=2.0 statsmodels>=0.14 scikit-learn>=1.4 matplotlib>=3.8`, pip-installable from repo root
- [X] T003 [P] Create `requirements-train.txt`: pinned fine-tuning deps — `unsloth`, `peft`, `transformers`, `torch`, `trl`, `bitsandbytes`, `accelerate` with versions locked
- [X] T004 [P] Create `.gitignore`: exclude `data/v1/*.jsonl`, `data/v2/*.jsonl`, `models/gguf/`, `models/adapters/`, `__pycache__/`, `*.egg-info/`, `.ipynb_checkpoints/`

---

## Phase 2: Foundational (Shared Data Structures)

**Purpose**: Core dataclasses and shared config that every subsequent phase depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Implement all dataclasses in `salestools/core.py`: `SalesFrame`, `DecompositionResult`, `AnomalyTable`, `GrowthMetrics`, `SegmentRanking` — field types and validation rules exactly as specified in `specs/001-sales-analyst-model/data-model.md`
- [X] T006 [P] Create `salestools/__init__.py`: import and re-export all 7 public functions (`load_sales`, `decompose_trend`, `growth_metrics`, `detect_anomalies`, `compare_segments`, `plot_annotated`, `narrate`) + `__version__ = "1.0.0"`
- [X] T007 [P] Implement `TrainingPair` dataclass + `to_jsonl()` / `from_jsonl()` helpers in `data/generator/schema.py` — fields and constraints from `specs/001-sales-analyst-model/contracts/training-pair-schema.md`
- [X] T008 [P] Create `training/config/system_prompt.txt`: fixed sales-analyst system prompt used by both fine-tune notebooks and `%%ask` magic (one paragraph, instructs model to generate only `salestools` code ≤15 lines ending with `narrate()`)
- [X] T009 [P] Create `training/config/lora_1.5b.yaml`: `rank=16`, `alpha=32`, `target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`, `bf16=true`, `batch_size=4`, `gradient_accumulation=4`, `max_seq_len=512`, `seed=42`

**Checkpoint**: `pip install -e .` succeeds; `python -c "import salestools"` imports without error.

---

## Phase 3: User Story 1 — Trend Question (Priority: P1) 🎯 MVP

**Goal**: Complete end-to-end pipeline — `salestools` library → data generator → 1.5B fine-tune → `%%ask` magic.

**Independent Test**: `%load_ext jupyter_magic` → `%%ask "is my overall trend up or down?"` → new cell inserted with ≤15-line `salestools` code → cell runs without edits → trend summary printed.

### Library (all 7 functions; all needed by the data generator)

- [X] T010 [US1] Implement `load_sales()` in `salestools/load.py`: CSV read, `datetime` parse, freq auto-detection from median gap, forward-fill gaps ≤3 periods (set `gap_flags=True`), flag gaps >3 periods, raise `ValueError` on parse failure or length < 4, raise `TypeError` on non-numeric value column
- [X] T011 [P] [US1] Implement `decompose_trend()` in `salestools/analysis.py`: STL via `statsmodels.tsa.seasonal.STL`, auto-period mapping (D→7, W→52, M→12, Q→4, Y→1); if series < 2×period print warning and return trend-only result; always return `DecompositionResult` with `fig`
- [X] T012 [P] [US1] Implement `growth_metrics()` in `salestools/analysis.py`: rolling period-over-period % change (configurable `window`), CAGR over full series, inflection points as sign-change dates in rolling growth; return `GrowthMetrics`
- [X] T013 [P] [US1] Implement `detect_anomalies()` in `salestools/anomalies.py`: `zscore` (threshold=3.0), `iqr` (multiplier=1.5), `iforest` (sklearn `IsolationForest`, contamination=0.05), `contextual` (z-score within day-of-week/month bucket, threshold=2.5), `auto` logic; return empty `AnomalyTable` when nothing found
- [X] T014 [P] [US1] Implement `compare_segments()` in `salestools/segments.py`: per-segment CAGR, `latest_value`, rolling slope `trend_direction` ('up'/'down'/'flat'); raise `ValueError` if no segment column; return `SegmentRanking` sorted by `ranked_by` descending
- [X] T015 [P] [US1] Implement `plot_annotated()` in `salestools/viz.py`: plot raw series, overlay `trend.trend` line if supplied, mark `anomalies.anomalies` dates as red points with `label` annotations; return `matplotlib.figure.Figure` (do not call `plt.show()`)
- [X] T016 [P] [US1] Implement `narrate()` in `salestools/narrate.py`: dispatch on result type; print plain-English summary for `AnomalyTable` (each anomaly with date + label), `GrowthMetrics` (trend direction, CAGR, inflection count), `SegmentRanking` (top and bottom segment), `DecompositionResult` (trend direction, seasonal strength); print "No notable findings." for empty results; never raise
- [X] T017 [US1] Validate `load_sales()` edge cases: create `tests/fixtures/gaps.csv` (3-period gap + >3-period gap) and `tests/fixtures/short.csv` (3-row file); assert gap-fill flags set correctly and `ValueError` raised for short series (depends on T010)

### Data Generator

- [X] T018 [P] [US1] Implement planted signal generators in `data/generator/signals.py`: `make_trend_up`, `make_trend_down`, `make_anomaly_spike`, `make_anomaly_drop`, `make_anomaly_contextual`, `make_segment_drag`, `make_scope_refusal_dataset` — each returns `(pd.DataFrame, detection_fn)` where `detection_fn(code_output) -> bool`
- [X] T019 [P] [US1] Implement question template library in `data/generator/questions.py`: ~20 canonical templates covering all signal types, each with 5 paraphrase variants; expose `get_questions(signal_type: str) -> list[str]`
- [X] T020 [US1] Implement multiprocessing sandbox verifier in `data/generator/verify.py`: `verify_pair(code: str, dataset: pd.DataFrame, detection_fn) -> (bool, str)` — spawn `multiprocessing.Process` with 30 s timeout, inject code into clean namespace containing only dataset + `salestools` imports, call `detection_fn` on namespace, return (passed, error_message) (depends on T007)
- [X] T021 [US1] Implement `data/generator/generate.py` CLI: flags `--salestools-version`, `--seed`, `--count`, `--split {train,held_out}`, `--delta-from <version>`, `--output <path>`; cross-product question templates × signal generators × seed range; filter via `verify.py`; write verified `TrainingPair` objects to JSONL (depends on T018, T019, T020, T007, all library functions T010–T016)
- [X] T022 [US1] Run generator to produce `data/v1/train.jsonl` (~1,000 pairs, seeds 0–8999) and `data/v1/held_out.jsonl` (~100 pairs, seeds 9000–9099); commit seed config to `data/seeds/v1.yaml`; confirm all pairs have `"verified": true` (depends on T021)

### Fine-Tune — 1.5B

- [X] T023 [US1] Implement `training/finetune_1.5b.ipynb`: load `data/v1/train.jsonl` → convert to ChatML format using `training/config/system_prompt.txt` → Unsloth QLoRA with `lora_1.5b.yaml` config targeting `Qwen/Qwen2.5-Coder-1.5B-Instruct` → train → save adapter to `models/adapters/1.5b/` (depends on T009, T022)
- [X] T024 [US1] Implement `training/export.sh`: parameterized by `MODEL_SIZE` and `ADAPTER_PATH`; merge LoRA adapter into base model → `llama.cpp convert_hf_to_gguf.py` → `llama-quantize q4_k_m` → write `Modelfile` → `ollama create sales-analyst-${MODEL_SIZE}` (depends on T023 for 1.5B)

### Jupyter Magic

- [X] T025 [US1] Implement `AskMagic` in `jupyter_magic/magic.py`: cell magic that posts `(system_prompt + "\n" + question)` to `http://localhost:11434/api/generate` with `stream=false`; strips response; calls `get_ipython().set_next_input(code, replace=False)` to insert code below; catches `ConnectionRefusedError` → inserts `# Ollama not running. Start with: ollama serve`
- [X] T026 [P] [US1] Create `jupyter_magic/__init__.py`: define `load_ipython_extension(ipython)` that registers `AskMagic`; expose `--model` (default `sales-analyst-1.5b`) and `--temperature` (default `0.1`) config options

**Checkpoint**: At this point, User Story 1 is fully functional and independently testable.

---

## Phase 4: User Story 2 — Anomaly Detection + Eval Harness (Priority: P2)

**Goal**: Validate that the 1.5B model correctly generates anomaly detection code; measure accuracy on held-out set.

**Independent Test**: `python eval/run_eval.py --model sales-analyst-1.5b --held-out data/v1/held_out.jsonl --signal-type anomaly_spike anomaly_drop anomaly_contextual` → EvalReport shows `signal_detection_accuracy` ≥ 0.80.

- [X] T027 [US2] Implement `eval/run_eval.py`: for each pair in `--held-out` JSONL, send question to `--model` via Ollama REST, execute returned code in sandbox via `data/generator/verify.py`, accumulate `pass_at_1`, `signal_detection_accuracy`, `scope_refusal_accuracy` (optionally filtered by `--signal-type`); write `EvalReport` JSON to `eval/reports/<variant>-<timestamp>.json`; print summary to stdout (depends on T020, T025 pattern)
- [X] T028 [US2] Run `python eval/run_eval.py --model sales-analyst-1.5b --held-out data/v1/held_out.jsonl --salestools-version 1.0.0`; confirm `pass_at_1 ≥ 0.85` and `signal_detection_accuracy ≥ 0.80` per SC-001/SC-002; commit report to `eval/reports/` (depends on T027, Phase 3 model)

**Checkpoint**: At this point, User Stories 1 and 2 should both work independently.

---

## Phase 5: User Story 4 — 3B Fine-Tune + A/B Comparison (Priority: P2)

**Goal**: Fine-tune the 3B variant on Colab Pro A100; generate side-by-side quality/speed comparison.

**Independent Test**: `python eval/compare.py eval/reports/1.5B-*.json eval/reports/3B-*.json` prints a table quantifying the quality gap between variants.

- [X] T029 [P] [US4] Create `training/config/lora_3b.yaml`: same hyperparams as `lora_1.5b.yaml`, targeting `Qwen/Qwen2.5-Coder-3B-Instruct`
- [X] T030 [US4] Implement `training/finetune_3b.ipynb`: same structure as `finetune_1.5b.ipynb`; load `data/v1/train.jsonl` → ChatML format → Unsloth QLoRA with `lora_3b.yaml` on A100 → save adapter to `models/adapters/3b/` (depends on T029, T022)
- [X] T031 [US4] Run 3B fine-tune on Colab Pro A100; run `training/export.sh MODEL_SIZE=3b ADAPTER_PATH=models/adapters/3b/` → `ollama create sales-analyst-3b`; write `models/gguf/sales-analyst-3b.meta.json` (depends on T030, T024)
- [X] T032 [P] [US4] Implement `eval/compare.py`: accept two EvalReport JSON file paths (1.5B and 3B); print aligned side-by-side table with columns `model_variant`, `pass_at_1`, `signal_detection_accuracy`, `scope_refusal_accuracy`, `eval_set_size`
- [X] T033 [US4] Run 3B eval + generate A/B report: `python eval/run_eval.py --model sales-analyst-3b ...`; then `python eval/compare.py eval/reports/1.5B-*.json eval/reports/3B-*.json`; confirm SC-003 satisfied (depends on T032, T027, T031)

**Checkpoint**: At this point, User Stories 1, 2, and 4 should all be independently functional.

---

## Phase 6: User Story 3 — Segment Comparison Validation (Priority: P3)

**Goal**: Validate that the 1.5B model correctly identifies underperforming segments.

**Independent Test**: `%%ask "which product is dragging sales down?"` on `tests/fixtures/multi_product.csv` → generated code identifies the planted underperformer.

- [X] T034 [US3] Create `tests/fixtures/multi_product.csv`: 3-product weekly dataset (52 weeks); product A CAGR +8%, product B CAGR +3%, product C CAGR −5% (planted underperformer); use seed=7000 for reproducibility
- [X] T035 [US3] Run segment validation: `python eval/run_eval.py --model sales-analyst-1.5b --held-out data/v1/held_out.jsonl --signal-type segment_drag`; confirm `signal_detection_accuracy ≥ 0.75`; manually run quickstart.md Phase 1 multi-product scenario to confirm `%%ask` identifies product C (depends on T027, T034)

**Checkpoint**: At this point, User Stories 1, 2, 3, and 4 should all be independently functional.

---

## Phase 7: User Story 5 — v2 Lifecycle Demo (Priority: P3)

**Goal**: Demonstrate versioned library evolution — v2 functions added, incremental fine-tune, before/after eval shows model adopts v2 API without v1 regression.

**Independent Test**: `python eval/compare.py eval/reports/1.5B-v1-*.json eval/reports/1.5B-v2-*.json` shows improvement on v2-function questions with no regression on v1-question accuracy.

- [X] T036 [P] [US5] Implement `salestools/forecast.py`: `forecast(sf: SalesFrame, horizon: int) -> ForecastResult`; use `statsmodels.tsa.holtwinters.ExponentialSmoothing`; `ForecastResult` holds `forecast_series: pd.Series`, `confidence_interval: pd.DataFrame`, `fig: plt.Figure`
- [X] T037 [P] [US5] Implement `salestools/cohort.py`: `cohort_analysis(sf: SalesFrame, cohort_col: str) -> CohortTable`; group by first-period bucket in `cohort_col`; `CohortTable` holds `retention: pd.DataFrame` (cohort × period), `fig: plt.Figure`
- [X] T038 [US5] Update `salestools/__init__.py`: export `forecast`, `cohort_analysis`; bump `__version__` to `"2.0.0"` (depends on T036, T037)
- [X] T039 [P] [US5] Add v2 signal generators to `data/generator/signals.py`: `make_forecast_question` (dataset with clear future trend), `make_cohort_question` (dataset with diverging cohort retention)
- [X] T040 [P] [US5] Add v2 question templates to `data/generator/questions.py`: 5 `forecast`-type templates × 5 paraphrases, 5 `cohort_analysis`-type templates × 5 paraphrases
- [X] T041 [US5] Generate delta dataset: `python data/generator/generate.py --salestools-version 2.0.0 --delta-from 1.0.0 --seed 42 --count 200 --output data/v2/delta.jsonl`; confirm all 200 pairs verified (depends on T039, T040, T038, T021)
- [X] T042 [US5] Implement `training/finetune_1.5b_v2.ipynb`: continue training from `models/adapters/1.5b/` checkpoint on `data/v2/delta.jsonl` only; save updated adapter to `models/adapters/1.5b-v2/` (depends on T041, T023)
- [X] T043 [US5] Export v2 model: `training/export.sh MODEL_SIZE=1.5b-v2 ADAPTER_PATH=models/adapters/1.5b-v2/` → `ollama create sales-analyst-1.5b-v2` (depends on T042, T024)
- [X] T044 [US5] Run before/after lifecycle eval: evaluate `sales-analyst-1.5b` and `sales-analyst-1.5b-v2` on both v1 and v2 held-out questions; `python eval/compare.py` to confirm v2 model improves on v2-function questions and v1-question accuracy within 2% of v1-only model; confirm SC-007 (depends on T043, T027, T032)

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Tests, hardening, and final validation.

- [X] T045 [P] Add unit tests for gap-fill, freq-detection, and error paths in `tests/unit/test_load.py` using fixtures from `tests/fixtures/` (depends on T017)
- [X] T046 [P] Add unit tests for `decompose_trend()` and `growth_metrics()` on a known 52-week series in `tests/unit/test_analysis.py`
- [X] T047 [P] Add unit tests for all 4 anomaly detection methods and empty-result case in `tests/unit/test_anomalies.py`
- [X] T048 [P] Add unit tests for `compare_segments()` ranking and `ValueError` on missing segment column in `tests/unit/test_segments.py`
- [X] T049 [P] Add unit tests for `narrate()` for each result type and "No notable findings." empty case in `tests/unit/test_narrate.py`
- [X] T050 Add integration smoke test in `tests/integration/test_generator_smoke.py`: run `generate.py --count 10 --seed 1` and assert 10 pairs returned, all `verified=true`, all `code` ≤15 lines (depends on T021)
- [X] T051 [P] Add `ModelArtifact` JSON generation to `training/export.sh`: write `models/gguf/<name>.meta.json` with `model_name`, `parameter_size`, `base_model`, `salestools_version`, `quantization`, `lora_rank`, `lora_alpha`, `training_seed`, `colab_hardware`, `training_minutes`, `gguf_path`
- [X] T052 Run quickstart.md end-to-end validation: execute all 6 phases sequentially; fix any gaps found; mark quickstart steps as verified

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational — builds entire pipeline
- **US2 (Phase 4)**: Depends on Phase 3 (model + generator sandbox) — eval harness
- **US4 (Phase 5)**: Depends on Phase 4 (`run_eval.py`) + Phase 3 data
- **US3 (Phase 6)**: Depends on Phase 4 (`run_eval.py`) + Phase 3 model
- **US5 (Phase 7)**: Depends on Phase 3 (base model) and Phase 4 (eval harness)
- **Polish (Phase 8)**: Depends on Phase 3 (library) and Phase 3 (generator)

### User Story Dependencies

- **US1 (P1)**: Requires Foundational — no other story dependencies
- **US2 (P2)**: Requires US1 (trained model + generator sandbox)
- **US4 (P2)**: Requires US2 (`run_eval.py`) + US1 training data
- **US3 (P3)**: Requires US2 (`run_eval.py`) + US1 trained model
- **US5 (P3)**: Requires US1 (base model + generator) + US2 (eval harness)

### Within Each Phase — Parallel Opportunities

- All `[P]`-marked tasks within a phase share no file dependencies and can run simultaneously
- Library functions T011–T016 are all `[P]` — can be implemented by 6 developers in parallel
- Signal generators (T018) and question templates (T019) are `[P]` — implement simultaneously
- 3B config (T029) and compare.py (T032) are `[P]` within Phase 5

---

## Parallel Example: Phase 3 Library Sub-Phase

```bash
# Launch all library function tasks simultaneously (different files, no deps):
Task: "T011 — Implement decompose_trend() in salestools/analysis.py"
Task: "T012 — Implement growth_metrics() in salestools/analysis.py"
Task: "T013 — Implement detect_anomalies() in salestools/anomalies.py"
Task: "T014 — Implement compare_segments() in salestools/segments.py"
Task: "T015 — Implement plot_annotated() in salestools/viz.py"
Task: "T016 — Implement narrate() in salestools/narrate.py"

# Then (after library complete), launch generator tasks simultaneously:
Task: "T018 — Implement signals.py"
Task: "T019 — Implement questions.py"
# T020 verify.py depends on T007 (schema) — run after T020 setup
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — CRITICAL, blocks everything
3. Complete Phase 3: US1
   - Library functions in parallel (T011–T016)
   - Generator (T018–T022 sequentially after library)
   - Fine-tune 1.5B on Colab Pro L4 (T023–T024)
   - Magic (T025–T026 in parallel with fine-tune wait)
4. **STOP and VALIDATE**: `%%ask "is my overall trend up or down?"` → code runs → summary printed
5. Demo if ready

### Incremental Delivery

1. Setup + Foundational → scaffold ready
2. US1 (Phase 3) → **MVP demo**: `%%ask` works for trend questions
3. US2 (Phase 4) → eval harness validates anomaly accuracy → EvalReport JSON produced
4. US4 (Phase 5) → 3B model trained → A/B comparison report produced
5. US3 (Phase 6) → segment comparison validated
6. US5 (Phase 7) → v2 lifecycle demo complete
7. Polish (Phase 8) → test suite, hardening, quickstart verified

### Parallel Team Strategy

With two developers after Foundational is complete:

- Developer A: US1 library functions (T011–T016 in parallel)
- Developer B: US1 data generator scaffold (T018–T019 in parallel)

Both converge at T020 (verifier needs both library + schema) → T021 (generator CLI) → T022 (data generation run)

Then:
- Developer A: Fine-tune 1.5B on Colab (T023–T024) while Developer B builds Jupyter magic (T025–T026)

---

## Notes

- `[P]` tasks = different files, no blocking dependencies within the phase
- `[USN]` label maps task to user story for traceability
- Phases 3–7 each produce a checkpoint independently verifiable against the quickstart.md
- Tests (Phase 8) are optional add-ons; the project is verified by execution throughout
- Avoid: same-file conflicts between parallel tasks, data/v1/ writes before T022 is complete
- Commit after each phase checkpoint; do not merge between phases without running the checkpoint
- **Constitution Note**: Principle III deviation (Colab Pro vs free T4) is documented in plan.md Complexity Tracking; consider amending constitution before starting Phase 3 fine-tune tasks
