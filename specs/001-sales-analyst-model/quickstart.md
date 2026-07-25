# Quickstart & Validation Guide: Local Sales-Analyst Codegen Model

**Date**: 2026-07-13 | **Plan**: [plan.md](plan.md)

This guide validates that each deliverable phase works end-to-end. Run each section in order;
each section has an independent pass/fail outcome.

---

## Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) for all Python environment/dependency
management — not bare `pip`. Install it once with `brew install uv` (macOS) or see the uv docs
for other platforms.

```bash
# Python 3.10+
uv run python --version

# Create the project venv and install salestools + dev deps (Phase 1 must be complete first)
uv venv
uv pip install -e ".[dev]"

# Verify install
uv run python -c "import salestools; print(salestools.__version__)"

# Ollama (for Phase 3+ validation only)
# Install from https://ollama.com (or `brew install ollama` on macOS), then: ollama serve
```

### llama.cpp + Ollama setup (required for `training/export.sh`)

After a Colab fine-tune finishes and the LoRA adapter zip has been downloaded and extracted into
`models/adapters/<size>/`, `training/export.sh` merges the adapter, converts it to GGUF, and
registers it with Ollama. This needs both tools set up locally first — they are **not** Python
packages and are not installed by `pip install -e .`.

```bash
# 1. Ollama — must be installed and running
#    macOS: brew install ollama   (or download from https://ollama.com)
ollama serve &          # leave running in the background
ollama --version         # sanity check

# 2. llama.cpp — cloned as a sibling directory to this repo and built
#    export.sh expects it at ../llama.cpp by default (override with LLAMA_CPP_DIR=/path/to/llama.cpp)
cd ..
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON   # macOS/Apple Silicon; drop -DGGML_METAL=ON on Linux/CUDA
cmake --build build --config Release -j
cd ../salestools-analyst

# 3. Verify export.sh can find both
"${LLAMA_CPP_DIR:-../llama.cpp}/build/bin/llama-quantize" --help | head -1
ollama list

# 4. Python deps for the merge + GGUF conversion steps (installed into the uv venv from above)
uv pip install torch transformers peft gguf sentencepiece "protobuf>=4.21.0,<5.0.0"
```

Once both are set up, run the export (see Phase 3 below):

```bash
MODEL_SIZE=1.5b ADAPTER_PATH=models/adapters/1.5b/ bash training/export.sh
```

---

## Phase 1: Validate `salestools` Library

**Goal**: Confirm all 7 public functions execute correctly on a minimal dataset.

```python
import pandas as pd
from salestools import (
    load_sales, decompose_trend, growth_metrics,
    detect_anomalies, compare_segments, plot_annotated, narrate
)

# Create minimal test CSV
import tempfile, os
df = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=52, freq="W"),
    "amount": [100 + i + (20 if i == 20 else 0) for i in range(52)]
})
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    df.to_csv(f, index=False)
    csv_path = f.name

# Load
sf = load_sales(csv_path, date_col="date", value_col="amount")
assert sf.freq == "W", f"Expected W, got {sf.freq}"
print(f"✅ load_sales: {len(sf.data)} rows, freq={sf.freq}")

# Decompose
result = decompose_trend(sf)
assert result.trend is not None
print(f"✅ decompose_trend: period={result.period}")

# Growth metrics
gm = growth_metrics(sf, window=4)
print(f"✅ growth_metrics: CAGR={gm.cagr:.1%}")

# Anomaly detection — week 20 is planted spike
at = detect_anomalies(sf, method="zscore")
assert len(at.anomalies) >= 1, "Planted spike should be detected"
print(f"✅ detect_anomalies: {len(at.anomalies)} anomaly(ies) found")

# Plot (no display in headless; just confirm Figure returned)
fig = plot_annotated(sf, anomalies=at, trend=result)
assert fig is not None
print("✅ plot_annotated: Figure returned")

# Narrate
narrate(at)   # should print a plain-English summary
print("✅ narrate: printed above")

os.unlink(csv_path)
```

**Expected output**: All ✅ lines printed; narrate prints a sentence about the anomaly.

---

## Phase 2: Validate Data Generator

**Goal**: Confirm the generator produces at least 10 verified pairs in < 5 minutes.

```bash
# From repo root
python data/generator/generate.py \
  --salestools-version 1.0.0 \
  --seed 42 \
  --count 10 \
  --output data/v1/smoke_test.jsonl

# Check output
python - <<'EOF'
import json
with open("data/v1/smoke_test.jsonl") as f:
    pairs = [json.loads(l) for l in f]

assert len(pairs) == 10, f"Expected 10, got {len(pairs)}"
assert all(p["verified"] for p in pairs), "All pairs must be verified"
assert all(len(p["code"].splitlines()) <= 15 for p in pairs), "Code ≤15 lines"
signal_types = {p["signal_type"] for p in pairs}
print(f"✅ Generator: {len(pairs)} pairs, signal types: {signal_types}")
EOF
```

**Expected output**: `✅ Generator: 10 pairs, signal types: {some set of signal_type values}`

---

## Phase 3: Validate Fine-Tuned Model (requires Ollama)

**Goal**: Confirm the `%%ask` cell magic reaches the local model and returns valid `salestools`
code.

**Prerequisites**: Both GGUF models loaded into Ollama:
```bash
ollama list  # should show sales-analyst-1.5b and sales-analyst-3b
```

```python
# In a Jupyter cell:
%load_ext jupyter_magic

# Test 1.5B
# %%ask --model sales-analyst-1.5b
# Is my overall sales trend up or down?
```

**Expected**: A new cell appears below with valid `salestools` code (≤15 lines, ending with
`narrate(...)`). Code runs without error when executed.

**Offline test**:
```bash
# Stop Ollama: pkill ollama
# Then in Jupyter: run %%ask
# Expected: friendly comment "# Ollama not running. Start with: ollama serve"
```

---

## Phase 4: Validate Evaluation Harness

**Goal**: Confirm eval harness runs against both model variants and produces EvalReport files.

```bash
# Run eval on held-out set (requires both models in Ollama)
python eval/run_eval.py \
  --model sales-analyst-1.5b \
  --held-out data/v1/held_out.jsonl \
  --salestools-version 1.0.0

python eval/run_eval.py \
  --model sales-analyst-3b \
  --held-out data/v1/held_out.jsonl \
  --salestools-version 1.0.0

# Compare
python eval/compare.py eval/reports/1.5B-*.json eval/reports/3B-*.json
```

**Expected**: Two JSON report files in `eval/reports/`; compare.py prints a side-by-side table.

**Success thresholds** (from [spec.md](spec.md) SC-001/SC-002):
- 1.5B: `pass_at_1` ≥ 0.85, `signal_detection_accuracy` ≥ 0.80

---

## Phase 5: End-to-End Setup Time Test

**Goal**: Validate that a fresh user can be up and running in < 10 minutes.

```bash
# Timer: start clock
time (
  pip install -e . &&
  ollama pull sales-analyst-1.5b &&
  jupyter notebook docs/demo.ipynb
)
```

**Expected**: Wall-clock time under 10 minutes (excluding initial `pip install` of heavy
fine-tuning deps, which are training-time only). The demo notebook should answer one sample
question end-to-end.

---

## Phase 6 (Phase 2 lifecycle): Validate v2 Incremental Fine-Tune

**Goal**: Confirm v2 eval shows improvement on `forecast` / `cohort_analysis` questions with no
v1 regression.

```bash
# Generate delta dataset (v2 functions only)
python data/generator/generate.py \
  --salestools-version 2.0.0 \
  --delta-from 1.0.0 \
  --seed 42 \
  --count 200 \
  --output data/v2/delta.jsonl

# Run v2 fine-tune (separate Colab notebook)

# Evaluate v1 vs v2 model on both v1 and v2 held-out questions
python eval/compare.py \
  eval/reports/1.5B-v1-*.json \
  eval/reports/1.5B-v2-*.json
```

**Expected**: v2 model scores higher on v2-specific questions; v1-question scores unchanged.
