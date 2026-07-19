# Implementation Plan: Local Sales-Analyst Codegen Model

**Branch**: `001-sales-analyst-model` | **Date**: 2026-07-13 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-sales-analyst-model/spec.md`

## Summary

Build a complete end-to-end pipeline: a `salestools` pip-installable Python library (~8 functions
for sales time-series analysis), a self-verifying synthetic data generator producing 800–1,500
execution-verified (question → `salestools` code) training pairs, QLoRA fine-tunes of both
Qwen2.5-Coder-1.5B-Instruct and 3B-Instruct on Colab Pro (L4/A100) exported to GGUF via
Ollama, a `%%ask` Jupyter cell magic for offline inference, and an evaluation harness that
produces a side-by-side A/B comparison of both model sizes on pass@1, signal-detection accuracy,
and scope-refusal accuracy on a held-out set.

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**:
- Library: `pandas ≥2.0`, `statsmodels ≥0.14`, `scikit-learn ≥1.4`, `matplotlib ≥3.8`
- Fine-tuning: `unsloth`, `peft`, `transformers`, `torch`, `trl`, `bitsandbytes`
- Export: `llama.cpp` (`convert_hf_to_gguf.py` + `llama-quantize`)
- Jupyter magic: `ipython`
- Inference: Ollama local REST (`localhost:11434`)
- Verification sandbox: Python standard library (`multiprocessing`)

**Storage**: Files only
- Training pairs: JSONL (`data/v1/train.jsonl`, `data/v1/held_out.jsonl`)
- Model artifacts: GGUF + metadata JSON (`models/gguf/`)
- LoRA configs: YAML (`training/config/`)
- Eval reports: JSON (`eval/reports/`)

**Testing**: `pytest` for `salestools` library unit + integration tests; execution-based
verification for training pairs; held-out set metrics for model quality

**Target Platform**:
- Training: Google Colab Pro (L4 for 1.5B, A100 for 3B)
- Inference: CPU-only laptop + Jupyter (via Ollama)

**Project Type**: Multi-component single-repo Python project
(library + data pipeline + training notebooks + Jupyter extension + eval harness)

**Performance Goals**:
- 1.5B model: well under 1 min per `%%ask` query on CPU-only laptop
- Data generation: verify 800–1,500 pairs with 30 s timeout per pair
- Training: ~45 min (1.5B on L4), ~90 min (3B on A100)

**Constraints**:
- No network calls at inference time (fully offline)
- Generated code ≤15 lines per pair; must end with `narrate(...)` or equivalent print
- Seeded, pinned-dep reproducibility for all training runs
- v1 data generator MUST be version-parameterized; not hardcoded to v1 function names
- Held-out seeds (9000–9999) never used in training data generation

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Library-First Code Generation | ✅ PASS | `salestools` built in Phase 1; all generated code targets only `salestools`; 15-line cap enforced (FR-008) |
| II. Self-Verifying Synthetic Data | ✅ PASS | FR-003/SC-006: 100% execution + signal-detection verification; unverified pairs discarded, never stored |
| III. Small, Local, Reproducible | ⚠️ DEVIATION | Constitution requires free T4 in ≤1 hour. Project uses Colab Pro (L4/A100) to enable 3B A/B variant. 1.5B alone meets the T4 constraint; 3B does not. Justified by explicit spec requirement (US-4/FR-005). Constitution amendment recommended — see Complexity Tracking. |
| IV. Execution-Based Evaluation | ✅ PASS | FR-006/SC-001/SC-002: pass@1, signal-detection, scope-refusal on held-out set with numeric thresholds; no vibes eval |
| V. Versioned Library Evolution | ✅ PASS | FR-011/FR-004: generator is version-parameterized (`--salestools-version`); v2 lifecycle spec'd (US-5, Phase 2) |
| VI. Notebook-Native UX | ✅ PASS | FR-007/FR-008/FR-009: `%%ask` cell magic; ≤15 lines; plain-English `narrate()` output; fully offline |

**Post-Phase 1 Re-check**: ✅ All gates pass after design. No new violations introduced by
data-model or contract decisions.

## Project Structure

### Documentation (this feature)

```text
specs/001-sales-analyst-model/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── salestools-api.md
│   ├── training-pair-schema.md
│   └── evalreport-schema.md
└── tasks.md             # Phase 2 output (/speckit-tasks command)
```

### Source Code (repository root)

```text
salestools/                     # Phase 1: pip-installable library
├── __init__.py                 # exports all public symbols + __version__
├── core.py                     # SalesFrame, DecompositionResult, AnomalyTable,
│                               #   GrowthMetrics, SegmentRanking dataclasses
├── load.py                     # load_sales()
├── analysis.py                 # decompose_trend(), growth_metrics()
├── anomalies.py                # detect_anomalies() — all four methods
├── segments.py                 # compare_segments()
├── viz.py                      # plot_annotated()
└── narrate.py                  # narrate()

data/
├── generator/                  # Phase 2: data generation pipeline
│   ├── signals.py              # planted signal generators (trend, spike, drop, contextual, segment)
│   ├── questions.py            # question template + paraphrase table (~20 templates × 5 phrasings)
│   ├── verify.py               # multiprocessing sandbox verifier (30 s timeout)
│   ├── schema.py               # TrainingPair dataclass + JSONL read/write
│   └── generate.py             # CLI entry point (--salestools-version, --seed, --count,
│                               #   --delta-from, --split train|held_out, --output)
├── v1/                         # generated dataset (gitignored for large files)
│   ├── train.jsonl
│   └── held_out.jsonl
└── seeds/                      # versioned seed config files

training/                       # Phase 3: fine-tuning notebooks + config
├── finetune_1.5b.ipynb         # Colab Pro (L4) notebook
├── finetune_3b.ipynb           # Colab Pro (A100) notebook
├── config/
│   ├── lora_1.5b.yaml          # rank=16, alpha=32, bf16, batch=4, grad_accum=4
│   ├── lora_3b.yaml            # same hyperparams; A100 target
│   └── system_prompt.txt       # fixed system prompt embedded in %%ask
└── export.sh                   # merge LoRA → GGUF q4_k_m → ollama create

models/
└── gguf/                       # gitignored; GGUF files + .meta.json live here locally

jupyter_magic/                  # Phase 4: %%ask IPython extension
├── __init__.py                 # register_magic(); load_ext entry point
└── magic.py                    # AskMagic — sends question to Ollama, inserts code cell

eval/                           # Phase 5: evaluation harness
├── run_eval.py                 # runs held-out set through Ollama, writes EvalReport JSON
├── compare.py                  # side-by-side A/B table for 1.5B vs 3B
└── reports/                    # EvalReport JSON files (gitignored for large batches)

tests/
├── unit/                       # salestools library function tests
│   ├── test_load.py
│   ├── test_analysis.py
│   ├── test_anomalies.py
│   ├── test_segments.py
│   └── test_narrate.py
└── integration/                # end-to-end: generate a small batch → verify → count
    └── test_generator_smoke.py
```

**Structure Decision**: Single-repo multi-component layout. Each top-level directory corresponds
to a deployment phase. `salestools/` is pip-installable from repo root (`pip install -e .`).
Data generator, training notebooks, and eval scripts are standalone and version-parameterized so
each phase can be run and validated independently, matching the constitution's Development
Workflow requirement.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|--------------------------------------|
| Colab Pro (L4/A100) instead of free T4 | 3B model variant is an explicit spec requirement (US-4/FR-005); 3B QLoRA fine-tuning cannot complete on T4 within the ~1-hour budget | Restricting to 1.5B-only would drop the A/B comparison requirement. 1.5B still targets ~45 min on L4 — well within the spirit of Principle III. **Recommended**: amend Principle III in constitution to "Colab Pro or equivalent (free T4 as floor for 1.5B-only runs)" with rationale. |
