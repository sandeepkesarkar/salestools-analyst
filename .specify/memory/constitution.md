<!--
## Sync Impact Report
- Version change: (uninitialized template) → 1.0.0
- Principles added:
  - I. Library-First Code Generation
  - II. Self-Verifying Synthetic Data (NON-NEGOTIABLE)
  - III. Small, Local, Reproducible
  - IV. Execution-Based Evaluation
  - V. Versioned Library Evolution
  - VI. Notebook-Native UX
- Sections added: Technical Constraints, Development Workflow, Governance
- Source: docs/constitution.md (ratified 2026-07-11)
- Templates requiring updates:
  - ✅ .specify/templates/plan-template.md — Constitution Check gate is generic and adapts dynamically; no structural change required
  - ✅ .specify/templates/spec-template.md — no mandatory sections added or removed
  - ✅ .specify/templates/tasks-template.md — no principle-driven task category changes required
  - ✅ .claude/skills/speckit-*/SKILL.md — no outdated agent-specific references found
- Follow-up TODOs: None; all fields populated from docs/constitution.md
-->

# SalesTools Analyst Model — Constitution

## Core Principles

### I. Library-First Code Generation
The fine-tuned model generates code **exclusively against the `salestools` library**, never raw
pandas/statsmodels pipelines. The library is the vocabulary; the model is the speaker. Any
capability the model should express MUST first exist as a `salestools` function. If a generated
solution cannot be written in ≤15 lines using `salestools`, the library API is wrong — fix the
library, not the training data.

### II. Self-Verifying Synthetic Data (NON-NEGOTIABLE)
Every training pair (question → code) MUST be verified by execution before entering the dataset:
1. The code MUST run without error in a clean kernel against a generated dataset.
2. The dataset MUST contain **planted** trends/anomalies, and the code's output MUST detect the
   planted signal (checked programmatically).

Unverified pairs are discarded, never hand-fixed. The data generator MUST be rerunnable and
versioned so that future `salestools` releases can generate delta datasets.

### III. Small, Local, Reproducible
- Base model: small coder model (≤3B params), fine-tuned with LoRA/QLoRA.
- Final artifact: GGUF served by Ollama, fully offline.
- Training MUST complete on a single free-tier Colab T4 (or equivalent consumer GPU) in under ~1 hour.
- All randomness seeded; dataset, LoRA config, and eval sets versioned in-repo.

### IV. Execution-Based Evaluation
Model quality is measured ONLY by executing its outputs: pass@1 (code runs) and detection
accuracy (planted anomaly/trend found) on a held-out set. No vibes-based eval. Regressions vs.
the previous checkpoint MUST block release.

### V. Versioned Library Evolution
`salestools` MUST follow semver. The project MUST support the full lifecycle: v1 API → fine-tune
→ v2 API (new functions) → incremental fine-tune on delta data → before/after eval confirming
the model adopts the new API. Nothing in v1 may make this lifecycle harder (e.g., data generator
hardcoded to v1 functions).

### VI. Notebook-Native UX
The end-user experience is Jupyter. A `%%ask` cell magic sends a natural-language question to
the local Ollama model and inserts runnable code into the next cell. Generated code MUST be
readable, short, and end with a printed plain-English summary of findings.

## Technical Constraints

- Python 3.10+; core deps: pandas, statsmodels, scikit-learn, matplotlib.
- Fine-tuning stack: Unsloth + PEFT (QLoRA), export via llama.cpp GGUF conversion.
- Base model default: Qwen2.5-Coder-1.5B-Instruct (3B as opt-in upgrade).
- No network calls at inference time; everything runs locally.

## Development Workflow

- Spec-driven via Spec Kit: spec → plan → tasks → implement.
- Phases ship independently: (1) salestools lib, (2) data generator, (3) fine-tune + export,
  (4) Jupyter magic, (5) eval harness, (6) v2 lifecycle demo.
- Each phase MUST have its own executable acceptance check before the next begins.

## Governance

This constitution supersedes ad-hoc decisions during implementation. Amendments require updating
this file with rationale and incrementing the version per semver rules. Any generated code,
dataset, or model that violates Principles I–IV MUST NOT be merged or released.

**Version**: 1.0.0 | **Ratified**: 2026-07-11 | **Last Amended**: 2026-07-13
