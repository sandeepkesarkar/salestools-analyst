#!/usr/bin/env bash
# Export a fine-tuned LoRA adapter to GGUF for Ollama.
#
# Usage:
#   MODEL_SIZE=1.5b ADAPTER_PATH=models/adapters/1.5b/ bash training/export.sh
#   MODEL_SIZE=3b   ADAPTER_PATH=models/adapters/3b/   bash training/export.sh
#
# Prerequisites:
#   - llama.cpp cloned at $LLAMA_CPP_DIR (default: ../llama.cpp)
#   - ollama installed and running (ollama serve)
#   - uv venv && uv pip install -e ".[dev]" torch transformers peft gguf sentencepiece "protobuf>=4.21.0,<5.0.0"

set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:?Set MODEL_SIZE=1.5b or 3b}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH=models/adapters/<size>/}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/../llama.cpp}"
CONFIG_PATH="${REPO_ROOT}/training/config/lora_${MODEL_SIZE}.yaml"

# Derived paths
MERGED_DIR="${REPO_ROOT}/models/merged/${MODEL_SIZE}"
GGUF_DIR="${REPO_ROOT}/models/gguf"
GGUF_UNQUANT="${GGUF_DIR}/sales-analyst-${MODEL_SIZE}-f16.gguf"
GGUF_QUANT="${GGUF_DIR}/sales-analyst-${MODEL_SIZE}-q4_k_m.gguf"
META_JSON="${GGUF_DIR}/sales-analyst-${MODEL_SIZE}.meta.json"
MODELFILE="${GGUF_DIR}/Modelfile.${MODEL_SIZE}"
OLLAMA_MODEL="sales-analyst-${MODEL_SIZE}"

mkdir -p "${MERGED_DIR}" "${GGUF_DIR}"

# ── Extract base model from config ────────────────────────────────────────────
BASE_MODEL=$(uv run python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG_PATH}'))
print(cfg['base_model'])
")
echo "Base model: ${BASE_MODEL}"
echo "Adapter:    ${ADAPTER_PATH}"

# ── Step 1: Merge LoRA adapter into base weights ──────────────────────────────
echo ""
echo "==> Step 1: Merging LoRA adapter into base model..."
uv run python3 - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("  Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "${BASE_MODEL}",
    torch_dtype=torch.float16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained("${BASE_MODEL}")

print("  Applying LoRA adapter...")
model = PeftModel.from_pretrained(model, "${ADAPTER_PATH}")
model = model.merge_and_unload()

print("  Saving merged model to ${MERGED_DIR}...")
model.save_pretrained("${MERGED_DIR}")
tokenizer.save_pretrained("${MERGED_DIR}")
print("  Done.")
EOF

# ── Step 2: Convert to GGUF (f16) ─────────────────────────────────────────────
echo ""
echo "==> Step 2: Converting to GGUF (f16)..."
uv run python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
    "${MERGED_DIR}" \
    --outfile "${GGUF_UNQUANT}" \
    --outtype f16

echo "  GGUF (f16): ${GGUF_UNQUANT}"

# ── Step 3: Quantize to q4_k_m ────────────────────────────────────────────────
echo ""
echo "==> Step 3: Quantizing to q4_k_m..."
"${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
    "${GGUF_UNQUANT}" \
    "${GGUF_QUANT}" \
    q4_k_m

echo "  GGUF (q4_k_m): ${GGUF_QUANT}"

# Remove f16 to save space
rm -f "${GGUF_UNQUANT}"

# ── Step 4: Write Modelfile ────────────────────────────────────────────────────
echo ""
echo "==> Step 4: Writing Modelfile..."
SYSTEM_PROMPT=$(cat "${REPO_ROOT}/training/config/system_prompt.txt")
cat > "${MODELFILE}" <<MODELEOF
FROM ${GGUF_QUANT}

SYSTEM """${SYSTEM_PROMPT}"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 256
PARAMETER stop "<|im_end|>"
MODELEOF

echo "  Modelfile: ${MODELFILE}"

# ── Step 5: Create Ollama model ────────────────────────────────────────────────
echo ""
echo "==> Step 5: Creating Ollama model '${OLLAMA_MODEL}'..."
ollama create "${OLLAMA_MODEL}" -f "${MODELFILE}"
echo "  Created: ${OLLAMA_MODEL}"

# ── Step 6: Write ModelArtifact JSON (T051) ────────────────────────────────────
echo ""
echo "==> Step 6: Writing ModelArtifact JSON..."
LORA_RANK=$(uv run python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_PATH}'))['lora']['r'])")
LORA_ALPHA=$(uv run python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_PATH}'))['lora']['lora_alpha'])")
SEED=$(uv run python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG_PATH}'))['training']['seed'])")
SALESTOOLS_VERSION=$(uv run python3 -c "import salestools; print(salestools.__version__)" 2>/dev/null || echo "1.0.0")
GGUF_SIZE=$(du -sh "${GGUF_QUANT}" 2>/dev/null | cut -f1 || echo "unknown")
CREATED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

uv run python3 - <<EOF
import json, os
meta = {
    "model_name": "${OLLAMA_MODEL}",
    "parameter_size": "${MODEL_SIZE}",
    "base_model": "${BASE_MODEL}",
    "salestools_version": "${SALESTOOLS_VERSION}",
    "quantization": "q4_k_m",
    "lora_rank": ${LORA_RANK},
    "lora_alpha": ${LORA_ALPHA},
    "training_seed": ${SEED},
    "colab_hardware": os.environ.get("COLAB_GPU", "unknown"),
    "training_minutes": None,
    "gguf_path": "${GGUF_QUANT}",
    "gguf_size": "${GGUF_SIZE}",
    "created_at": "${CREATED_AT}",
}
with open("${META_JSON}", "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
EOF

echo ""
echo "✅ Export complete!"
echo "   Ollama model : ${OLLAMA_MODEL}"
echo "   GGUF file    : ${GGUF_QUANT}"
echo "   Meta JSON    : ${META_JSON}"
echo ""
echo "Test with:"
echo "   ollama run ${OLLAMA_MODEL} \"Is my overall sales trend going up?\""
