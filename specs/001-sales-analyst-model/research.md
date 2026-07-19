# Research: Local Sales-Analyst Codegen Model

**Phase**: 0 | **Date**: 2026-07-13 | **Plan**: [plan.md](plan.md)

## Decision 1: Sandboxed Pair Verification

**Decision**: Use Python `multiprocessing.Process` with a 30-second wall-clock timeout to execute
each (question → code) pair in an isolated subprocess. Each run gets a clean global namespace
containing only the generated dataset and the `salestools` imports; signal detection is checked
programmatically after execution by inspecting the subprocess return value.

**Rationale**: Self-contained, zero extra infrastructure, fast enough for 1,500 pairs (≈12 min
serial; parallelizable). The code to be verified uses only `salestools` and standard library, so
there is no meaningful sandbox escape risk.

**Alternatives considered**:
- Docker container per pair — isolated but setup overhead is high; overkill for trusted synthetic
  code that only calls `salestools` functions.
- `nbformat` + Jupyter kernel execution — supports realistic notebook context but is slower to
  spin up (kernel startup ~2s × 1,500 pairs = 50 min); subprocess approach is 10–20× faster.
- `RestrictedPython` — would require maintaining a policy for every `salestools` function call;
  fragile and unnecessary given trusted synthetic code.

## Decision 2: QLoRA Configuration (Unsloth + PEFT)

**Decision**:
- Both variants use rank=16, alpha=32, target modules = `q_proj, k_proj, v_proj, o_proj,
  gate_proj, up_proj, down_proj`, bf16 precision.
- 1.5B: batch_size=4, gradient_accumulation=4, max_seq_len=512, ~45 min on Colab Pro L4.
- 3B: batch_size=4, gradient_accumulation=4, max_seq_len=512, ~90 min on Colab Pro A100.
- Chat format: Qwen's built-in ChatML template (already in the tokenizer).
- Training data format: `{"role":"user","content":"<question>"}` / `{"role":"assistant",
  "content":"<code>"}` pairs in JSONL.

**Rationale**: rank=16 is the standard starting point for code generation tasks (good expressivity
without overfitting on <2k examples). Qwen2.5-Coder already understands Python; the fine-tune
teaches `salestools` vocabulary, not general coding, so a lightweight adapter suffices.
Unsloth's 2× speed gain over vanilla PEFT is material on Colab's billed compute.

**Alternatives considered**:
- Full fine-tune — unnecessary parameter count for vocabulary adaptation; would blow Colab VRAM.
- rank=64 — higher expressivity but ~4× VRAM; risk of overfitting on 1,500 pairs.
- GPT-4-style data-augmentation for more pairs — violates reproducibility and self-containedness
  principles; synthetic generator is the authoritative source.

## Decision 3: Ollama REST Integration Pattern for `%%ask`

**Decision**: The `%%ask` IPython cell magic sends a single non-streaming POST to
`http://localhost:11434/api/generate` with `{"model": "<model_name>", "prompt": "<system>+<user
question>", "stream": false}`. The response `.response` field is stripped and inserted as a new
code cell below the `%%ask` cell using `get_ipython().set_next_input()`. Ollama connectivity is
checked with a 2-second socket timeout; `ConnectionRefusedError` surfaces a friendly
`# Ollama not running. Start with: ollama serve` comment in the output.

**Rationale**: Non-streaming keeps the magic simple (one requests call, no async). The system
prompt is fixed and embedded in the magic module so users never need to craft it. `set_next_input`
is the standard IPython idiom for inserting generated code without auto-running it.

**Alternatives considered**:
- Streaming response — more complex; no UX benefit for short (<15 line) code outputs.
- LangChain/LlamaIndex wrapper — heavy dependency; the Ollama REST API is three lines.
- Auto-execute the generated cell — explicitly out of scope (spec FR-007, US assumption).

## Decision 4: Trend Decomposition Library

**Decision**: `statsmodels.tsa.seasonal.STL` (Seasonal-Trend decomposition using LOESS) for
`decompose_trend()`. Frequency is auto-detected from the `DatetimeIndex` (daily→7, weekly→52,
monthly→12); falls back to `period=2` with a warning if the series is too short for a full cycle.

**Rationale**: STL handles irregular seasonal amplitudes, is robust to anomalies in the trend
component (unlike classical additive decomposition), and is already a `salestools` dependency
(statsmodels). The `period` parameter is user-overridable for non-standard data.

**Alternatives considered**:
- `statsmodels` classical decomposition — less robust; anomalies bleed into trend estimate.
- Prophet — excellent but adds a heavy optional dependency and is overkill for the narrow use
  case; also no longer well-maintained.
- Rolling mean — too simple; cannot separate seasonality from trend reliably.

## Decision 5: GGUF Quantization Format

**Decision**: `q4_k_m` for both model variants. Export path: merge LoRA adapter into base model
→ `llama.cpp` `convert_hf_to_gguf.py` → `llama-quantize output.gguf q4_k_m` → `ollama create`.

**Rationale**: `q4_k_m` is the community-standard quantization for CPU inference: ~4.0 bits/param
average, good quality retention for code generation (empirically ~1–2% pass@1 drop vs. fp16
on 1–3B models), and small enough for the 1.5B variant to fit comfortably in 1–2 GB RAM.

**Alternatives considered**:
- `q5_k_m` — ~20% larger file, marginal quality gain; does not change the demo story.
- `q8_0` — ~2× larger than `q4_k_m`; slower CPU inference; unnecessary given model size.
- `q2_k` — aggressive compression degrades code quality noticeably at these parameter counts.

## Decision 6: Question Template Paraphrase Strategy

**Decision**: Maintain a static paraphrase table: each of ~20 canonical question templates has
3–5 manually-authored phrasings (synonyms, tone variants). The generator cross-products templates
with paraphrasings and dataset seeds. No LLM is used for paraphrase generation.

**Rationale**: Keeps the generator fully self-contained and reproducible (no external API calls,
no random LLM outputs). 20 templates × 5 paraphrases × varied seeds → 1,500+ candidates before
filtering; the verification step prunes to the quality bar, not the generation step.

**Alternatives considered**:
- LLM-based paraphrase augmentation — creates a dependency on an external service, breaks
  reproducibility, and is unnecessary given the template count.
- Single phrasing per template — limits lexical diversity and risks the model memorizing surface
  form rather than semantics.

## Decision 7: Held-Out Set Isolation

**Decision**: The held-out set (~100 pairs) is drawn from a reserved range of dataset seeds
(seeds 9000–9999) that are never used during training data generation. Seeds 0–8999 are available
for training. The split is enforced at generation time via a `--split held_out` flag that activates
the reserved seed range.

**Rationale**: Seed-based split is deterministic, reproducible, and requires no post-hoc shuffle
or stratification logic. Using a disjoint seed range (rather than a random fraction of generated
pairs) guarantees that the held-out datasets are genuinely unseen — different planted signals,
different question phrasings — not just code held back from the same underlying dataset.

**Alternatives considered**:
- Random 10% split of generated pairs — risks the same planted signal appearing in both train
  and held-out (since signal type is the stratifying variable, not the pair index).
- Manual curation of held-out pairs — not reproducible and not version-aware.
