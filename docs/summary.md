# salestools-analyst — Project Summary

> A locally-run AI assistant that answers questions about your sales data — no cloud, no
> subscription, no data leaving your machine.

---

## The Problem in Plain English

A small-business owner sits down with a spreadsheet of daily sales figures and wants to know:
*"Is my business trending up or down? Were there any unusual weeks? Which product is dragging
performance down?"*

Today, answering those questions requires either hiring an analyst, copying sensitive data into
a cloud AI service, or learning statistics tools that take months to master. None of those
options are fast, cheap, or private.

This project builds a purpose-built AI assistant — running entirely on your own computer — that
can read your sales CSV and answer those questions instantly in plain English.

---

## What You Get

At the end of this project, a user opens a Jupyter notebook (a standard data science tool),
loads their sales file, and types a question in plain English:

```
%%ask
Which weeks this year had unusual sales, and is my overall trend up or down?
```

The assistant generates a short analysis script in the next cell. Running it produces:

- An annotated chart with anomalies marked in red
- A plain-English summary: *"Week 23 was 3.1× above expected — likely a promo spike.
  Underlying trend: +4% monthly growth after removing seasonality."*

Everything runs offline. No data is sent anywhere. It works on a standard laptop.

---

## How It Works — The Simple Version

Think of it like training a specialist. A general-purpose AI knows a little about everything.
This project takes a small, efficient AI model and teaches it one specific skill: turning
plain-English questions about sales into working analysis code.

To teach it, the project:

1. **Builds the vocabulary first.** Creates a library of analysis functions tailored to sales
   data — things like "detect anomalies" and "compare product trends."

2. **Creates thousands of practice examples.** Automatically generates pairs of questions and
   correct answers (analysis code), then checks every single one actually works before using it
   for training.

3. **Trains the specialist.** Fine-tunes a small AI model on those verified examples so it
   learns the exact vocabulary and patterns needed.

4. **Packages it for everyday use.** Connects the trained model to Jupyter notebooks so analysts
   can use it with a simple `%%ask` command, with no internet required.

5. **Measures how well it works.** Runs the trained model against a set of test questions it
   has never seen, checks the answers, and produces a scorecard.

---

## The Six Building Blocks

### 1. The `salestools` Library — *The Vocabulary*

Before training any AI, this project defines the exact set of analysis tools the AI is allowed
to use. Think of it as a specialist's toolbox with eight named tools:

| Tool | What it does |
|------|-------------|
| `load_sales` | Reads a sales CSV; handles missing dates and irregular data automatically |
| `decompose_trend` | Separates the underlying trend from seasonal patterns and noise |
| `growth_metrics` | Calculates growth rates and finds turning points in the data |
| `detect_anomalies` | Flags unusual values using four different statistical methods |
| `compare_segments` | Ranks products or regions by their growth performance |
| `plot_annotated` | Draws a chart with anomalies marked and labelled in red |
| `narrate` | Converts analysis results into a printed plain-English sentence |

**Why define the toolbox first?** The AI is only ever allowed to write code using these tools.
If the AI cannot answer a question in 15 lines of `salestools` code, that is a signal to improve
the toolbox — not to let the AI write complex raw statistics code that would be hard to read or
audit.

---

### 2. The Data Generator — *The Practice Examples*

Training an AI requires thousands of examples of correct behaviour. Creating those by hand would
take months and would be inconsistent. Instead, this project builds a program that generates
them automatically — and, critically, *checks every single one before accepting it.*

Here is how a single training example is created:

1. A dataset is created with a **planted signal** — a known trend, anomaly, or underperforming
   product hidden inside realistic-looking sales numbers.
2. A natural-language question is paired with a `salestools` code answer that should detect that
   signal.
3. The code is executed in an isolated environment. Two checks must pass:
   - The code runs without any errors.
   - The code's output actually finds the planted signal.
4. Only if both checks pass does this example enter the training set. Failed examples are
   discarded — never patched by hand.

The result is a dataset of 800–1,500 verified question-and-answer pairs, each one proven to
work. The held-out test set (100 examples, never used in training) is kept separate from day one
using a different range of random seeds.

---

### 3. The Fine-Tuning Pipeline — *Teaching the Specialist*

The project starts with an existing small AI model
(`Qwen2.5-Coder`, a model designed for writing code) and teaches it the `salestools` vocabulary
by training it on the verified examples above.

This process — called *fine-tuning* — is much faster and cheaper than training a model from
scratch. It runs on a cloud GPU (Google Colab Pro) and takes 45–90 minutes.

**Two sizes are trained and compared:**

| Model | Size | Training time | Best for |
|-------|------|--------------|---------|
| 1.5B | ~1 GB on disk | ~45 min on L4 GPU | CPU-only laptops; fast answers |
| 3B | ~2 GB on disk | ~90 min on A100 GPU | Higher accuracy; GPU available |

After training, the model is compressed into a compact format (GGUF) and loaded into
[Ollama](https://ollama.com) — a free tool that runs AI models locally on your machine, similar
to how a web server runs a website.

---

### 4. The Jupyter Magic — *The User Interface*

The user-facing feature is a single Jupyter cell magic: `%%ask`. When a user types a question
after `%%ask` and runs the cell, the magic:

1. Sends the question to the locally-running model (no internet needed).
2. Receives back a short `salestools` script.
3. Inserts that script into a new cell directly below.
4. Waits for the user to review and run it — the code is never auto-executed.

If the local model is not running, the cell shows a friendly setup hint rather than a
cryptic error message.

---

### 5. The Evaluation Harness — *The Scorecard*

After training, the model is tested against the 100 held-out examples it has never seen.
Three metrics are measured:

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **pass@1** | Does the generated code run without errors? | ≥ 85% |
| **Signal-detection accuracy** | Does the code find the planted signal in the test data? | ≥ 80% |
| **Scope-refusal accuracy** | Does the model correctly decline out-of-scope questions? | Measured |

Both the 1.5B and 3B models are evaluated side by side so the quality-versus-speed tradeoff is
quantified in a concrete report — not guessed at.

---

### 6. The v2 Lifecycle Demo — *Proving the System Grows*

One of the core goals of this project is to demonstrate that a fine-tuned specialist model can
be updated when the underlying library evolves — without starting from scratch.

In Phase 2 of the project, two new functions are added to `salestools` (`forecast` and
`cohort_analysis`). The data generator reruns on those new functions only, producing a small
*delta* dataset. The model is then fine-tuned for a short additional session on that delta data.

A before-and-after evaluation confirms:
- The updated model uses the new functions correctly for new question types.
- The updated model still works correctly for all the original question types.

This demonstrates the full lifecycle: **build → train → ship → extend → retrain → verify**.

---

## Design Principles (For the Curious)

Six rules govern every decision in this project:

1. **Library-first.** The AI writes `salestools` code only. If it cannot answer in 15 lines
   using the library, the library needs a new function — not a workaround.

2. **No unverified training data.** Every training example must pass two automated checks
   (runs cleanly + detects the planted signal) before it can be used. There are no exceptions.

3. **Small, local, reproducible.** Models ≤ 3B parameters. Fully offline at inference.
   Training reproducible with a fixed random seed and pinned software versions.

4. **Execution-based evaluation only.** Model quality is measured by running its outputs —
   not by human opinion or subjective assessment.

5. **Versioned evolution.** The library follows semantic versioning. The training pipeline is
   designed from the start to support incremental updates without full retraining.

6. **Notebook-native.** The end user experience lives entirely in Jupyter. Generated code is
   short, readable, and ends with a plain-English summary.

---

## Project Phases at a Glance

```
Phase 1  →  Build the salestools library
Phase 2  →  Build the data generator; produce 1,000+ verified training pairs
Phase 3  →  Fine-tune 1.5B model; build %%ask Jupyter magic
Phase 4  →  Evaluate 1.5B model; build eval harness
Phase 5  →  Fine-tune 3B model; produce A/B comparison report
Phase 6  →  Validate segment-comparison capability
Phase 7  →  Add v2 library functions; incremental fine-tune; lifecycle demo
Phase 8  →  Tests, hardening, end-to-end validation
```

Each phase produces a concrete, independently testable deliverable before the next one begins.

---

## Technical Specifications (For Technical Readers)

| Dimension | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.10+ | Standard for data science tooling |
| Base models | Qwen2.5-Coder-1.5B-Instruct / 3B-Instruct | Strong code generation at small size |
| Fine-tuning method | QLoRA via Unsloth + PEFT | Efficient; fits Colab Pro VRAM |
| QLoRA config | rank=16, alpha=32, bf16, batch=4 | Balanced expressivity vs. overfitting risk on ~1k examples |
| Training hardware | Colab Pro L4 (1.5B) / A100 (3B) | Reproducible, accessible, no local GPU required |
| Model export | GGUF q4_k_m via llama.cpp | Best quality/size ratio for CPU inference |
| Inference runtime | Ollama (local REST at localhost:11434) | Zero-install local model serving |
| Decomposition | statsmodels STL | Robust to anomalies in the trend component |
| Anomaly detection | z-score / IQR / IsolationForest / contextual | Four methods; auto-selects based on series type |
| Sandbox execution | Python multiprocessing, 30 s timeout | Fast, no Docker needed for trusted synthetic code |
| Training pair format | JSONL (ChatML) | Native format for Unsloth / PEFT training |
| Held-out split | Seed range 9000–9999 (disjoint from training) | Guarantees no signal overlap between train and test |
| Evaluation metrics | pass@1 / signal-detection / scope-refusal | Execution-based; no subjective scoring |

---

## In One Sentence

**salestools-analyst** is a self-contained pipeline that trains a small, offline AI model to
answer plain-English questions about sales data — then proves it works by running its outputs
against a test set it has never seen.
