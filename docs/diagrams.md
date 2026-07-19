# salestools-analyst — Architecture Diagrams

Four views of the system, read in order:

1. **Pipeline Overview** — all six phases and how artifacts flow between them
2. **Data Generation** — how one verified training pair is produced
3. **`%%ask` Runtime** — what happens when a user types a question in Jupyter
4. **Evaluation & A/B Comparison** — how both model sizes are assessed

---

## 1. Pipeline Overview

```mermaid
flowchart TD
    subgraph P1["Phase 1 · salestools Library"]
        direction LR
        LS[load_sales]
        DT[decompose_trend]
        GM[growth_metrics]
        DA[detect_anomalies]
        CS[compare_segments]
        PA[plot_annotated]
        NA[narrate]
    end

    subgraph P2["Phase 2 · Data Generator"]
        direction TB
        SIG["signals.py\n────────\nmake_trend_up/down\nmake_anomaly_spike/drop\nmake_anomaly_contextual\nmake_segment_drag\nmake_scope_refusal"]
        QT["questions.py\n────────\n~20 templates\n× 5 paraphrases each"]
        VER["verify.py\n────────\nsubprocess sandbox\n30 s timeout\ndetection_fn check"]
        GEN["generate.py CLI\n────────\n--salestools-version\n--seed  --count\n--split  --delta-from"]
        TRAIN[("data/v1/train.jsonl\n~1,000 verified pairs\nseeds 0–8999")]
        HELD[("data/v1/held_out.jsonl\n~100 verified pairs\nseeds 9000–9099")]
    end

    subgraph P3A["Phase 3A · Fine-Tune 1.5B  (Colab Pro L4 ~45 min)"]
        direction LR
        NB1["finetune_1.5b.ipynb\nUnsloth + QLoRA\nrank=16 / bf16 / batch=4"]
        EX1["export.sh\nmerge adapter\n→ GGUF q4_k_m\n→ ollama create"]
        OL1[("Ollama\nsales-analyst-1.5b")]
    end

    subgraph P3B["Phase 3B · Fine-Tune 3B  (Colab Pro A100 ~90 min)"]
        direction LR
        NB3["finetune_3b.ipynb\nUnsloth + QLoRA\nrank=16 / bf16 / batch=4"]
        EX3["export.sh\nmerge adapter\n→ GGUF q4_k_m\n→ ollama create"]
        OL3[("Ollama\nsales-analyst-3b")]
    end

    subgraph P4["Phase 4 · Jupyter Magic"]
        MAG["jupyter_magic\n────────\n%%ask cell magic\nPOST localhost:11434\nset_next_input()"]
    end

    subgraph P5["Phase 5 · Eval Harness"]
        RE["run_eval.py\n────────\npass@1\nsignal_detection_accuracy\nscope_refusal_accuracy"]
        CP["compare.py\n────────\nside-by-side A/B table\n1.5B vs 3B"]
        RPT[("eval/reports/\n*.json")]
    end

    subgraph P6["Phase 6 · v2 Lifecycle  (Phase 2 of project)"]
        V2L["salestools v2\n+ forecast()\n+ cohort_analysis()"]
        V2G["generate.py\n--delta-from 1.0.0\n--salestools-version 2.0.0"]
        V2F["finetune_1.5b_v2.ipynb\nincremental fine-tune\non delta dataset only"]
        V2E["before/after eval\nv1 model vs v2 model\nno v1 regression"]
    end

    %% library feeds generator
    P1 -->|"salestools functions\nused in code templates\n& for verification"| SIG
    P1 -->|"imported by\ngenerated code"| VER

    %% generator internal flow
    SIG --> GEN
    QT  --> GEN
    GEN --> VER
    VER -->|"verified=true pairs"| TRAIN
    VER -->|"verified=true pairs\n(reserved seeds)"| HELD

    %% training
    TRAIN --> NB1
    TRAIN --> NB3
    NB1   --> EX1 --> OL1
    NB3   --> EX3 --> OL3

    %% runtime
    OL1 --> MAG

    %% eval
    OL1  --> RE
    OL3  --> RE
    HELD -->|"never seen in training"| RE
    RE   --> RPT
    RPT  --> CP

    %% v2 lifecycle
    P1    -.->|"extend"| V2L
    GEN   -.->|"--delta-from"| V2G
    V2L   --> V2G
    V2G   --> V2F
    OL1   -.->|"base adapter"| V2F
    V2F   --> V2E
    RE    -.->|"reused"| V2E
```

---

## 2. Data Generation — One Verified Training Pair

```mermaid
sequenceDiagram
    actor Dev as Developer
    participant CLI as generate.py CLI
    participant SIG as signals.py
    participant QT  as questions.py
    participant VER as verify.py
    participant SBX as Subprocess Sandbox
    participant LIB as salestools
    participant OUT as train.jsonl / held_out.jsonl

    Dev->>CLI: python generate.py<br/>--salestools-version 1.0.0<br/>--seed 42 --count 1000<br/>--split train

    loop For each (signal_type, paraphrase_id, seed) in cross-product
        CLI->>SIG: make_{signal_type}(seed)
        SIG-->>CLI: (DataFrame with planted signal,<br/>detection_fn)

        CLI->>QT: get_questions(signal_type)[paraphrase_id]
        QT-->>CLI: question string<br/>(e.g. "Which weeks had unusual sales?")

        Note over CLI: look up code template<br/>for this signal_type<br/>(hand-written salestools code)

        CLI->>VER: verify_pair(code, DataFrame, detection_fn)

        VER->>SBX: spawn Process(target=run_code,<br/>timeout=30s)

        SBX->>LIB: import salestools
        SBX->>SBX: exec(code, clean_namespace)

        alt code raises exception
            SBX-->>VER: error: "NameError: ..."
            VER-->>CLI: (passed=False, error=msg)
            Note over CLI: discard pair — never stored
        else code runs cleanly
            SBX->>SBX: detection_fn(namespace_output)
            alt planted signal NOT detected
                SBX-->>VER: (passed=False, "signal not found")
                VER-->>CLI: (passed=False, error=msg)
                Note over CLI: discard pair — never stored
            else planted signal detected
                SBX-->>VER: (passed=True, "")
                VER-->>CLI: (passed=True, "")
                CLI->>OUT: write TrainingPair JSON line<br/>{ id, question, code,<br/>  signal_type, verified: true,<br/>  salestools_version, dataset_seed,<br/>  paraphrase_id }
            end
        end
    end

    CLI-->>Dev: ✅ 1,000 verified pairs written<br/>0 unverified pairs stored
```

---

## 3. `%%ask` Runtime — User Asks a Question in Jupyter

```mermaid
sequenceDiagram
    actor User as Analyst (Jupyter)
    participant IPY  as IPython Kernel
    participant MAG  as AskMagic<br/>(jupyter_magic)
    participant OLL  as Ollama<br/>localhost:11434
    participant MDL  as sales-analyst-1.5b<br/>(GGUF in RAM)
    participant LIB  as salestools library
    participant PLT  as matplotlib

    User->>IPY: Run cell:<br/>%%ask<br/>"Which weeks had unusual sales?"

    IPY->>MAG: dispatch to AskMagic.run()<br/>cell_body = "Which weeks had unusual sales?"

    MAG->>MAG: read model config<br/>(default: sales-analyst-1.5b)
    MAG->>MAG: load system_prompt.txt

    MAG->>OLL: POST /api/generate<br/>{ model: "sales-analyst-1.5b",<br/>  prompt: system_prompt + question,<br/>  stream: false }

    alt Ollama not running
        OLL-->>MAG: ConnectionRefusedError
        MAG->>IPY: set_next_input(<br/>  "# Ollama not running.<br/>   Start with: ollama serve")
        IPY-->>User: 🔴 hint cell inserted below
    else Ollama running
        OLL->>MDL: tokenise + run inference<br/>(q4_k_m, CPU or GPU)
        MDL-->>OLL: generated token stream

        Note over MDL,OLL: model generates ≤15 lines<br/>of salestools code ending<br/>with narrate(...)

        OLL-->>MAG: { "response": "<code string>" }

        MAG->>MAG: strip whitespace / markdown fences
        MAG->>IPY: set_next_input(code, replace=False)
        IPY-->>User: 🟢 new cell inserted below<br/>(not auto-executed)

        User->>IPY: ▶ Run generated cell

        IPY->>LIB: sf = load_sales('sales.csv')
        LIB-->>IPY: SalesFrame (validated, freq=W)

        IPY->>LIB: anomalies = detect_anomalies(sf)
        LIB-->>IPY: AnomalyTable (3 rows)

        IPY->>LIB: plot_annotated(sf, anomalies=anomalies)
        LIB->>PLT: draw series + red markers
        PLT-->>User: 📊 annotated chart displayed

        IPY->>LIB: narrate(anomalies)
        LIB-->>User: "Week 23 was 3.1σ above expected —<br/>likely promo spike.<br/>Weeks 8 and 41 were drops."
    end
```

---

## 4. Evaluation & A/B Comparison

```mermaid
flowchart TD
    HO[("held_out.jsonl\n~100 pairs\nsignal_types:\nanomaly_spike\nanomaly_drop\nanomaly_contextual\ntrend_up / trend_down\nsegment_drag\nscope_refusal")]

    subgraph EVAL1["run_eval.py — 1.5B"]
        direction TB
        Q1["send question\nto sales-analyst-1.5b\nvia Ollama"]
        R1["receive code\nresponse"]
        S1["execute code\nin sandbox\n(verify.py, 30s)"]
        D1{"outcome"}
        P1A["pass@1 += 1\ndetect planted signal\n→ signal_detection += 1"]
        P1B["scope_refusal\ncheck\n→ scope_refusal += 1"]
        P1C["fail: discard"]
        RP1[("eval/reports/\n1.5B-timestamp.json\n────────\npass_at_1: 0.88\nsignal_detection: 0.83\nscope_refusal: 0.95")]
    end

    subgraph EVAL3["run_eval.py — 3B"]
        direction TB
        Q3["send question\nto sales-analyst-3b\nvia Ollama"]
        R3["receive code\nresponse"]
        S3["execute code\nin sandbox"]
        D3{"outcome"}
        P3A["pass@1 += 1\nsignal_detection += 1"]
        P3B["scope_refusal\ncheck"]
        P3C["fail: discard"]
        RP3[("eval/reports/\n3B-timestamp.json\n────────\npass_at_1: 0.91\nsignal_detection: 0.87\nscope_refusal: 0.96")]
    end

    subgraph CMP["compare.py — A/B Report"]
        TBL["┌──────────┬─────────┬────────────┬──────────────┐
│ variant  │ pass@1  │ sig_detect │ scope_refus  │
├──────────┼─────────┼────────────┼──────────────┤
│ 1.5B     │  0.88   │   0.83     │     0.95     │
│ 3B       │  0.91   │   0.87     │     0.96     │
└──────────┴─────────┴────────────┴──────────────┘
→ 3B gains +3pp pass@1, +4pp signal_detection
→ 1.5B preferred for CPU inference latency"]
    end

    HO --> Q1
    HO --> Q3

    Q1 --> R1 --> S1 --> D1
    D1 -->|"ran + signal found"| P1A
    D1 -->|"scope_refusal pair"| P1B
    D1 -->|"error or\nsignal missed"| P1C
    P1A --> RP1
    P1B --> RP1

    Q3 --> R3 --> S3 --> D3
    D3 -->|"ran + signal found"| P3A
    D3 -->|"scope_refusal pair"| P3B
    D3 -->|"error or\nsignal missed"| P3C
    P3A --> RP3
    P3B --> RP3

    RP1 --> TBL
    RP3 --> TBL

    TBL --> DEC{"Decision"}
    DEC -->|"latency priority\nor CPU-only"| USE1["Deploy 1.5B\nas default %%ask model"]
    DEC -->|"quality priority\nwith GPU"| USE3["Deploy 3B\nfor power users"]
```
