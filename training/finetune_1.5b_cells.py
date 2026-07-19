# ── CELL 1: Install dependencies ──────────────────────────────────────────────
import os

!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q sentencepiece pyyaml

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# After this cell finishes: Runtime → Restart session, then continue from Cell 2


# ── CELL 2: Imports ────────────────────────────────────────────────────────────
import unsloth

import json, os, subprocess
from pathlib import Path
import yaml
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── CELL 3: Paths and config ───────────────────────────────────────────────────
REPO_ROOT   = Path("/content/salestools-analyst")
CONFIG_PATH = REPO_ROOT / "training/config/lora_1.5b.yaml"
PROMPT_PATH = REPO_ROOT / "training/config/system_prompt.txt"
TRAIN_DATA  = REPO_ROOT / "data/v1/train.jsonl"
ADAPTER_OUT = REPO_ROOT / "models/adapters/1.5b"
ADAPTER_OUT.mkdir(parents=True, exist_ok=True)

if not REPO_ROOT.exists():
    subprocess.run(["git", "clone", "https://github.com/sandeepkesarkar/salestools-analyst.git", str(REPO_ROOT)], check=True)

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
SYSTEM_PROMPT = Path(PROMPT_PATH).read_text().strip()

print("Config:", cfg)
print(f"System prompt: {SYSTEM_PROMPT[:80]}...")


# ── CELL 4: Generate training data ────────────────────────────────────────────
# Skip if train.jsonl already exists
import subprocess
if not TRAIN_DATA.exists():
    subprocess.run(
        ["python", "data/generator/generate.py", "--count", "1000", "--seed", "42", "--output", str(TRAIN_DATA)],
        cwd=str(REPO_ROOT), check=True
    )
else:
    print("train.jsonl already exists, skipping generation.")


# ── CELL 5: Load training data ─────────────────────────────────────────────────
CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n{assistant}<|im_end|>"
)

records = []
with open(TRAIN_DATA) as f:
    for line in f:
        pair = json.loads(line.strip())
        if not pair.get("verified", False):
            continue
        records.append({"text": CHATML_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            user=pair["question"],
            assistant=pair["code"],
        )})

dataset = Dataset.from_list(records)
print(f"Loaded {len(dataset)} verified pairs")
print("Sample:\n", dataset[0]["text"][:300])


# ── CELL 6: Load base model ────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["base_model"],
    max_seq_length=cfg["max_seq_len"],
    dtype=None,
    load_in_4bit=True,
)
print("Base model loaded.")


# ── CELL 7: Apply LoRA ────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_rank"],
    target_modules=cfg["target_modules"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=cfg["seed"],
)
print("LoRA applied.")
model.print_trainable_parameters()


# ── CELL 8: Train ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=cfg["max_seq_len"],
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation"],
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=cfg["seed"],
        output_dir=str(ADAPTER_OUT / "checkpoints"),
        save_steps=100,
        save_total_limit=2,
        report_to="none",
    ),
)
trainer_stats = trainer.train()
print(f"Done. Runtime: {trainer_stats.metrics.get('train_runtime', 0):.0f}s")


# ── CELL 9: Save adapter ──────────────────────────────────────────────────────
model.save_pretrained(str(ADAPTER_OUT))
tokenizer.save_pretrained(str(ADAPTER_OUT))
print(f"Saved to {ADAPTER_OUT}")
print("Files:", list(ADAPTER_OUT.glob("*")))


# ── CELL 10: Download adapter ─────────────────────────────────────────────────
import shutil
from google.colab import files

shutil.make_archive("/content/adapter_1.5b", "zip", str(ADAPTER_OUT))
files.download("/content/adapter_1.5b.zip")
