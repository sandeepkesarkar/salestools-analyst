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
        FC["forecast (v2)"]
        CH["cohort_analysis (v2)"]
    end

    subgraph P2["Phase 2 · Data Generator"]
        direction TB
        SIG["signals.py\n────────\nmake_trend_up/down\nmake_anomaly_spike/drop\nmake_anomaly_contextual\nmake_segment_drag\nmake_scope_refusal\nmake_forecast_question (v2)\nmake_cohort_question (v2)"]
        QT["questions.py\n────────\n~20 templates\n× 5 paraphrases each"]
        VER["verify.py\n────────\nsubprocess sandbox\n30 s timeout\nreturns (ran_ok, detected_ok, error)"]
        GEN["generate.py CLI\n────────\n--salestools-version\n--seed  --count\n--split  --delta-from\n--signal-types"]
        TRAIN[("data/v1/train.jsonl\n~1,000 verified pairs\nseeds 0–8999")]
        HELD[("data/v1/held_out.jsonl\n~100 verified pairs\nseeds 9000–9999")]
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

    subgraph P6["Phase 6 · v2 Lifecycle"]
        V2L["salestools v2\n+ forecast()\n+ cohort_analysis()"]
        V2G["generate.py\n--delta-from 1.0.0\n--salestools-version 2.0.0"]
        V2D[("data/v2/delta.jsonl\n~200 verified pairs\nseed 42 (training data)")]
        V2H[("data/v2/held_out.jsonl\n40 verified pairs\nseeds 9000+ (disjoint —\nnever used in training)")]
        REP[("data/v2/replay_v1.jsonl\n~210 pairs, 30/type\nsampled from v1 train.jsonl")]
        V2F["finetune_1.5b_v2.ipynb\nincremental fine-tune on\ndelta + replay (~1:1 mix)\nprevents catastrophic forgetting"]
        V2E["before/after eval\nv1 model vs v2 model,\non BOTH held-out sets\nno v1 regression"]
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
    P1     -.->|"extend"| V2L
    GEN    -.->|"--delta-from"| V2G
    V2L    --> V2G
    V2G    -->|"seed 42\n(train range)"| V2D
    V2G    -->|"seed 9000+\n(held-out range)"| V2H
    TRAIN  -->|"sample_replay.py\n30/signal_type"| REP
    V2D    --> V2F
    REP    --> V2F
    OL1    -.->|"base adapter\n(continue training,\nnot a fresh LoRA)"| V2F
    V2F    --> V2E
    V2H    -->|"v2 questions:\ndid it learn them?"| V2E
    HELD   -->|"v1 questions:\ndid it forget them?"| V2E
    RE     -.->|"reused"| V2E
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
            VER-->>CLI: (ran_ok=False, detected_ok=False, error=msg)
            Note over CLI: discard pair — never stored
        else code runs cleanly
            SBX->>SBX: detection_fn(namespace_output)
            alt planted signal NOT detected
                SBX-->>VER: (ran ok, but signal not found)
                VER-->>CLI: (ran_ok=True, detected_ok=False, error=msg)
                Note over CLI: discard pair — never stored<br/>(generator requires BOTH ok;<br/>eval harness tracks them separately)
            else planted signal detected
                SBX-->>VER: (ran ok, signal found)
                VER-->>CLI: (ran_ok=True, detected_ok=True, "")
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

        MAG->>MAG: dedent code<br/>(no fence-stripping needed —<br/>system prompt forbids markdown fences)
        MAG->>IPY: set_next_input(code, replace=False)
        IPY-->>User: 🟢 new cell inserted below<br/>(not auto-executed)

        User->>IPY: ▶ Run generated cell

        IPY->>LIB: sf = load_sales('sales.csv')
        LIB-->>IPY: SalesFrame (validated,<br/>freq=W-SUN — anchored to<br/>the data's own weekday)

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
    HO[("held_out.jsonl\n100 pairs, seeds 9000–9999\nsignal_types:\nanomaly_spike\nanomaly_drop\nanomaly_contextual\ntrend_up / trend_down\nsegment_drag\nscope_refusal")]

    subgraph EVAL1["run_eval.py — 1.5B"]
        direction TB
        Q1["send question\nto sales-analyst-1.5b\nvia Ollama"]
        R1["receive code\nresponse"]
        S1["execute code\nin sandbox\n(verify.py, 30s)"]
        D1{"outcome"}
        P1A["ran_ok → pass@1 += 1\ndetected_ok → signal_detection += 1\n(tracked separately;\nboth excl. scope_refusal pairs)"]
        P1B["scope_refusal pair:\ncorrect refusal?\n→ scope_refusal += 1"]
        P1C["crashed: neither\ncounter increments"]
        RP1[("eval/reports/\nsales-analyst-1.5b-<ts>.json\n────────\npass_at_1: 1.00\nsignal_detection: 1.00\nscope_refusal: 1.00")]
    end

    subgraph EVAL3["run_eval.py — 3B"]
        direction TB
        Q3["send question\nto sales-analyst-3b\nvia Ollama"]
        R3["receive code\nresponse"]
        S3["execute code\nin sandbox"]
        D3{"outcome"}
        P3A["ran_ok → pass@1 += 1\ndetected_ok → signal_detection += 1"]
        P3B["scope_refusal pair:\ncorrect refusal?\n→ scope_refusal += 1"]
        P3C["crashed: neither\ncounter increments"]
        RP3[("eval/reports/\nsales-analyst-3b-<ts>.json\n────────\npass_at_1: 1.00\nsignal_detection: 1.00\nscope_refusal: 1.00")]
    end

    subgraph CMP["compare.py — A/B Report"]
        TBL["┌──────────┬─────────┬────────────┬──────────────┐
│ variant  │ pass@1  │ sig_detect │ scope_refus  │
├──────────┼─────────┼────────────┼──────────────┤
│ 1.5B     │  1.00   │   1.00     │     1.00     │
│ 3B       │  1.00   │   1.00     │     1.00     │
└──────────┴─────────┴────────────┴──────────────┘
→ Zero delta on every metric AND every individual signal type
→ 1.5B deployed as the %%ask default: identical measured quality on
  this eval set, for a fraction of the memory/latency cost"]
    end

    HO --> Q1
    HO --> Q3

    Q1 --> R1 --> S1 --> D1
    D1 -->|"non-refusal pair,\nran without error"| P1A
    D1 -->|"scope_refusal pair"| P1B
    D1 -->|"exception raised"| P1C
    P1A --> RP1
    P1B --> RP1

    Q3 --> R3 --> S3 --> D3
    D3 -->|"non-refusal pair,\nran without error"| P3A
    D3 -->|"scope_refusal pair"| P3B
    D3 -->|"exception raised"| P3C
    P3A --> RP3
    P3B --> RP3

    RP1 --> TBL
    RP3 --> TBL

    TBL --> DEC{"Decision"}
    DEC -->|"this eval set shows\nno quality difference"| USE1["Deploy 1.5B\nas default %%ask model\n(3B available for harder\nfuture eval sets)"]
```
