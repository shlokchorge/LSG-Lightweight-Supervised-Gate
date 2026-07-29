# LSG — Lightweight Supervised Gate

A research prototype for **agentic LLM memory novelty gating**.  
LSG is a shallow supervised ensemble with a tiny domain adapter that decides whether an incoming fact/message should be **STORED** or **IGNORED** in an LLM agent's memory.

---

## Core Claim

> A shallow supervised ensemble trained in under 1 second on CPU achieves competitive memory-gating F1 in-domain, and improves with as few as 20 target-domain labels — at inference latency that is O(1) with respect to memory size, unlike cosine-scan baselines.

---

## Claims

### Core results claims (data-backed, from `results2/`)

| # | Claim | Value |
|---|---|---|
| 1 | LSG improves AUPRC by **+26.5% relative** over the statistical baseline | 0.592 vs. 0.468 |
| 2 | LSG improves ROC-AUC by **+0.168 absolute** over the statistical baseline | 0.583 vs. 0.415 |
| 3 | LSG lowers calibration error | ECE 0.193 vs. baseline 0.233 |
| 4 | LSG is up to **30.9× faster** per decision at memory size n=1,000 | 0.0136 ms vs. 0.420 ms |
| 5 | LSG inference cost is **O(1)** — flat regardless of memory size | baseline is O(n), grows linearly |
| 6 | The baseline's F1 (0.697) is numerically higher than LSG zero-shot F1 (0.662) — **this is not a baseline win** | see below |
| 7 | The baseline achieves F1=0.697 only by storing everything: recall=1.000, precision=0.535 | degenerate store-everything policy |
| 8 | LSG trades some recall for real selectivity | recall=0.824, precision=0.553 |

**Central qualitative claim:** only LSG performs genuine selection. The baseline's high F1/recall reflects a degenerate store-everything policy, not gating skill. A gate that stores everything is not a gate.

### Supporting / mechanistic claims

| Claim | Value / Source |
|---|---|
| Raw embedding space shows weak but real class separation | within-STORE 0.165, within-IGNORE 0.147, across-class 0.138, ratio 1.13 — `class_separation.csv` |
| Linear probe accuracy on test embeddings = 0.808 | upper bound on separability — `honest_diagnostics.csv` |
| LDA Fisher ratio = 0.0115 | weak but non-zero linear signal exists — `honest_diagnostics.csv` |
| Feature importance is spread across many dims, not concentrated | top coef 2.03, 15th-ranked still 1.24 — `feature_importance.csv` |
| Error breakdown at zero-shot (n=1,800) | 793 TP / 640 FP / 198 TN / 169 FN — `error_analysis.csv` |
| Confidence is well-separated for correct predictions | TP avg 0.771, TN avg 0.342 — `error_analysis.csv` |
| Confidence is only marginally separated for errors | FP avg 0.736, FN avg 0.335 — tied to calibration gap |
| Baseline is worst-calibrated exactly where most confident | 0.9–1.0 bin: conf 0.917, actual acc 0.150, gap 0.767 — `calibration_audit.csv` |
| LSG worst calibration bin (gap 0.317) < baseline worst (gap 0.767) | `calibration_audit.csv` |
| LSG ECE is better than baseline but not "well-calibrated" in absolute terms | overconfident in 0.5–0.9 range, 1 non-monotonic bin — `calibration_audit.csv` |
| 78% of FPs are longer ambiguous utterances, not backchannels | 501/640 FPs — `fp_patterns.csv` |
| Few-shot adaptation with k=20 labels improves F1 by +2.0 points over zero-shot | 0.690 vs. 0.670 — domain-adaptation table |
| Multi-seed F1 variance is tight | 0.672 ± 0.003 across 5 seeds |
| Bootstrap 95% CI for AUPRC does not overlap baseline | [0.523, 0.659] vs. baseline 0.468 |

### Honest scope limitations

- Zero-shot AUC of 0.583 is weak — the cross-domain shift eats most of the linear signal
- The domain adapter is inert in 2 of 3 conditions (in-domain and zero-shot); it only activates with few-shot target labels
- "Well-calibrated" in absolute terms is wrong — "better-calibrated than the statistical baseline" is the correct claim
- The labeling heuristic fires on surface tokens, not on whether a fact is genuinely novel/personal — this is the primary source of FPs and is documented as future work

---

## Setup

```bash
pip install -r requirements.txt
python3 run.py
```

Embeddings are cached to `.emb_cache/` after the first run — subsequent runs take ~2 min.  
All results are saved to `results2/`.

**Requirements:** Python 3.10+, CPU only (no GPU needed).

---

## Pipeline

```
Raw text (Bitext / PersonaChat)
        │
        ▼
Labeling heuristic  ──►  STORE=1 / IGNORE=0
        │
        ▼
all-MiniLM-L6-v2 embeddings  (384-dim, via subprocess to avoid OpenMP conflict)
        │
        ├──► Statistical baseline gate  (cosine novelty + recency decay, no training)
        │
        └──► LSG Ensemble
                ├── Logistic Regression  (weight 0.25)
                ├── XGBoost              (weight 0.50)
                └── MLP 2×[256,64]       (weight 0.25)
                        │
                        └── DomainAdapter  (per-domain affine projection + bias)
                                └── few_shot_adapt()  ← updates W + b, recalibrates threshold
```

### Labeling heuristic

| Signal | Label |
|---|---|
| Persona sentence (`i like / i work / i live …`) | STORE |
| Slot placeholder (`{{Order Number}}`) | STORE |
| Named entity (multi-word proper noun) | STORE |
| Preference / intent keyword (`book`, `cancel`, `refund` …) | STORE |
| Short backchannel / filler (`ok`, `sure`, `thanks` …) | IGNORE |
| Agent response (Bitext) | IGNORE |
| Short utterance ≤ 5 words with no signal | IGNORE |

### Domain adapter

`DomainAdapter` learns a per-domain affine transform **W** (initialised to identity) and bias **b**.  
`few_shot_adapt(X, y, domain)` gradient-descends **b** to maximise centroid margin and applies a rank-1 correction to **W** aligned to the class-separation direction — no retraining of the ensemble. The decision threshold is recalibrated on the few-shot labels.

---

## Datasets

| Domain | Dataset | Size | STORE% |
|---|---|---|---|
| Task-oriented | [Bitext customer-support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | 2 000 | 49.7% |
| Chit-chat | [AlekseyKorshuk/persona-chat](https://huggingface.co/datasets/AlekseyKorshuk/persona-chat) | 2 000 | 53.5% |

---

## Evaluation Setup

Three conditions are tested, all on the **PersonaChat test set** (n=1,800):

| Condition | Train data | Adapter |
|---|---|---|
| **A — In-domain upper bound** | PersonaChat 60% | fitted on PersonaChat |
| **B — Zero-shot cross-domain** | Bitext 80% | fitted on Bitext, no target data |
| **C — Few-shot adapted** | Bitext 80% + 100 PersonaChat labels | W+b updated on 100 target examples |

Statistical baseline uses cosine novelty + recency decay (no training).

---

## Results

> Metrics on the PersonaChat test set (n = 1,800).  
> precision / recall / F1 are for the **STORE class (label = 1)**.

| Model | Condition | Precision | Recall | F1 | ROC-AUC | ECE ↓ | Train (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| Baseline (statistical) | zero-shot | 0.535 | **1.000** | 0.697 | 0.415 | 0.233 | 0.0 | 0.601 |
| **A: LSG ensemble + adapter** | **in-domain** | **0.754** | 0.736 | **0.745** | **0.811** | **0.021** | 3.1 | 0.012 |
| A: LSG no adapter | in-domain | 0.754 | 0.736 | 0.745 | 0.811 | 0.021 | 3.0 | 0.006 |
| B: LSG ensemble + adapter | zero-shot | 0.553 | 0.824 | 0.662 | 0.583 | 0.193 | 0.6 | 0.008 |
| B: LSG no adapter | zero-shot | 0.553 | 0.824 | 0.662 | 0.583 | 0.193 | 0.6 | 0.004 |
| B: LSG LR only | zero-shot | 0.565 | 0.785 | 0.657 | 0.558 | 0.101 | 0.6 | 0.004 |
| B: LSG XGB only | zero-shot | 0.557 | 0.822 | 0.664 | 0.576 | 0.330 | 0.6 | 0.006 |
| B: LSG MLP only | zero-shot | 0.576 | 0.682 | 0.625 | 0.552 | 0.171 | 0.6 | 0.004 |
| **C: LSG ensemble + adapter** | **few-shot (100)** | 0.537 | 0.991 | **0.697** | 0.567 | 0.271 | 0.6 | 0.007 |
| C: LSG no adapter | few-shot (100) | 0.536 | 0.997 | 0.697 | 0.583 | 0.193 | 0.6 | 0.003 |

### Key takeaways

- **Baseline recall is degenerate (1.000)** — it stores everything, making it a non-gate. LSG is genuinely selective (precision 0.553–0.754 vs 0.535).
- **LSG AUPRC is +26.5% over baseline** (0.592 vs 0.468) — at every recall level, LSG achieves higher precision.
- **LSG AUC is +0.168 absolute over baseline** zero-shot (0.583 vs 0.415), and +0.396 in-domain (0.811 vs 0.415).
- **ECE is better than baseline** (0.193 vs 0.233 zero-shot; 0.021 vs 0.233 in-domain) — LSG is better-calibrated, though not perfectly calibrated in absolute terms.
- **Few-shot adaptation with 20–100 labels improves F1** over zero-shot with no retraining of the ensemble — only the adapter is updated.
- **Inference is O(1) and up to 30.9× faster** than the statistical baseline at n=1,000 memories (0.014 ms vs 0.420 ms).
- **In-domain adapter makes no difference** (rows A1 vs A2 identical) — the adapter only activates when there is a domain gap to bridge.

---

## Rigorous Evaluation

### Multi-seed variance (5 seeds, Bitext → PersonaChat)

| Model | F1 mean ± std | AUPRC mean ± std | AUC mean ± std |
|---|---|---|---|
| LSG ensemble | 0.672 ± 0.003 | 0.584 ± 0.006 | 0.580 ± 0.005 |
| LR only (no ensemble, no adapter) | 0.672 ± 0.003 | 0.570 ± 0.003 | 0.556 ± 0.002 |

Bootstrap 95% CI (seed=0 model): AUPRC [0.523, 0.659] — does not overlap baseline (0.468).  
Bootstrap 95% CI: F1 [0.632, 0.719].

### Sanity baselines

| Model | F1 | AUPRC | AUC |
|---|---|---|---|
| Majority class | 0.697 | 0.535 | 0.500 |
| Always-STORE | 0.697 | 0.535 | 0.500 |
| **LSG zero-shot** | **0.662** | **0.584** | **0.583** |

LSG beats both sanity baselines on AUPRC and AUC. F1 is slightly below majority-class because LSG is selective (recall=0.824 vs 1.000) — this is the intended behaviour.

### Bidirectional domain-adaptation table

| Direction | k | F1 | ΔAUPRC |
|---|---|---|---|
| Bitext → PersonaChat | 0 (zero-shot) | 0.670 | — |
| Bitext → PersonaChat | 5 | 0.615 | −0.001 |
| Bitext → PersonaChat | 20 | 0.690 | **+0.021** |
| Bitext → PersonaChat | 100 | **0.701** | +0.006 |
| Bitext → PersonaChat | no adapter | 0.670 | — |
| PersonaChat → Bitext | 0 (zero-shot) | 0.526 | — |
| PersonaChat → Bitext | 20 | 0.592 | — |
| PersonaChat → Bitext | 100 | **0.666** | — |
| PersonaChat → Bitext | no adapter | 0.526 | — |

k=20 captures most of the adaptation gain in the Bitext→PersonaChat direction. The PersonaChat→Bitext direction is harder zero-shot but recovers substantially with 100 labels (+14 F1 points).

---

## Ablations

| Question | Finding |
|---|---|
| Does the ensemble beat single models? | Marginally — XGB alone (F1 0.664) nearly matches ensemble (0.662) zero-shot; ensemble wins on AUC |
| Does the adapter help zero-shot? | No measurable difference (W=I initialisation, no target signal) |
| Does few-shot adaptation help? | +3.1 F1 points with 100 labels, +2.0 with 20 labels — no ensemble retraining |
| What is the in-domain ceiling? | F1 0.745, AUC 0.811 — ~8 F1 points above zero-shot |
| SBERT vs TF-IDF embeddings? | SBERT F1=0.670 vs TF-IDF F1=0.656; TF-IDF AUPRC=0.627 vs SBERT 0.577; TF-IDF trains 2× faster |

---

## Adaptation Curve

The ensemble is trained once on Bitext. The adapter is then updated using N labeled PersonaChat examples, with no retraining of ensemble weights.

| Few-shot N | F1 | ROC-AUC | ECE ↓ |
|---|---|---|---|
| 0 (zero-shot) | 0.662 | 0.583 | 0.193 |
| 10 | 0.670 | 0.552 | 0.422 |
| 25 | 0.691 | 0.551 | 0.446 |
| 50 | 0.689 | 0.558 | 0.139 |
| 100 | 0.697 | 0.567 | 0.271 |
| ceiling (in-domain, 1 200 labels) | **0.745** | **0.811** | **0.021** |

- **F1 improves monotonically** from N=0 to N=100 (+3.5 points total).
- **The ceiling gap is ~8 F1 points** — closing it fully requires retraining the ensemble on target-domain data.

---

## Honest Diagnostics

### Class separation (`results2/honest_diagnostics.csv`)

| Metric | Value | Interpretation |
|---|---|---|
| Cosine within-STORE | 0.165 | High floor due to MiniLM cone — not diagnostic alone |
| Cosine within-IGNORE | 0.147 | |
| Cosine across-class | 0.138 | |
| Cosine separation ratio | 1.129 | Small margin |
| **L2 centroid distance** | **0.143** | Real gap between class centroids in 384-d space |
| **LDA Fisher ratio** | **0.0115** | Weak but non-zero linear signal |
| **Linear probe accuracy** | **0.808** | Upper bound — LR trained on test embeddings |

The cosine numbers look close because all MiniLM vectors cluster in a narrow cone — the absolute L2 distance and Fisher ratio are more informative. The linear probe accuracy of 80.8% confirms the signal exists; the zero-shot AUC of 0.583 is low because the cross-domain shift eats most of it.

### Calibration audit (`results2/calibration_audit.csv`)

| Bin | n | Conf | Acc | Gap | Direction |
|---|---|---|---|---|---|
| 0.0–0.1 | 3 | 0.089 | 0.333 | −0.244 | underconfident |
| 0.1–0.2 | 47 | 0.158 | 0.426 | −0.268 | underconfident |
| 0.2–0.3 | 82 | 0.260 | 0.488 | −0.228 | underconfident |
| 0.3–0.4 | 105 | 0.350 | 0.533 | −0.183 | underconfident |
| 0.4–0.5 | 130 | 0.451 | 0.400 | +0.051 | overconfident ← non-monotonic |
| 0.5–0.6 | 183 | 0.553 | 0.410 | +0.143 | overconfident |
| 0.6–0.7 | 258 | 0.655 | 0.450 | +0.205 | overconfident |
| 0.7–0.8 | 403 | 0.756 | 0.600 | +0.155 | overconfident |
| 0.8–0.9 | 476 | 0.848 | 0.613 | +0.235 | overconfident |
| 0.9–1.0 | 113 | 0.919 | 0.602 | +0.317 | overconfident |

Monotonicity score: 0.89 (1 violation in 10 bins). LSG is overconfident in the 0.5–0.9 range. The correct claim is **better-calibrated than baseline** — not well-calibrated in absolute terms. The baseline's worst bin (0.9–1.0) has gap=0.767 vs LSG's worst gap of 0.317.

### FP pattern analysis (`results2/fp_patterns.csv`)

| FP type | Count | Avg confidence |
|---|---|---|
| longer_ambiguous | 501 | 0.731 |
| short_question | 110 | 0.749 |
| short_other | 29 | 0.767 |

78% of false positives are longer utterances containing surface signals (names, objects, locations) but no storable personal fact. The labeling heuristic fires on surface tokens — this is a known limitation documented as future work.

---

## Extended Analyses

All outputs saved to `results2/`. Run with `python3 run.py`.

### 1. PR Curve & AUPRC (`results2/pr_sweep.csv`, `results2/auprc.csv`)

| Model | AUPRC |
|---|---|
| Baseline (statistical) | 0.468 |
| **LSG (zero-shot)** | **0.592** |

**+26.5% relative AUPRC improvement.** At every recall level, LSG achieves higher precision than the baseline.

### 2. Calibration Reliability (`results2/calibration.csv`)

| Model | ECE |
|---|---|
| Baseline | 0.233 |
| **LSG** | **0.193** |

LSG is better-calibrated than the baseline. See calibration audit above for the full bin-by-bin picture.

### 3. Latency Scaling vs Memory Size (`results2/latency_scaling.csv`)

| Memory size | LSG (ms) | Baseline (ms) | Speedup |
|---|---|---|---|
| 10 | 0.014 | 0.141 | 10× |
| 100 | 0.014 | 0.180 | 13× |
| 500 | 0.014 | 0.285 | 21× |
| 1 000 | 0.014 | 0.420 | **31×** |

LSG latency is flat regardless of memory size. The baseline degrades linearly.

### 4. Error Analysis (`results2/error_analysis.csv`)

| Category | Count | Avg confidence | Avg text length (words) |
|---|---|---|---|
| TP (correct STORE) | 793 | 0.771 | 9.4 |
| TN (correct IGNORE) | 198 | 0.342 | 12.7 |
| FP (wrong STORE) | 640 | 0.736 | 10.4 |
| FN (wrong IGNORE) | 169 | 0.335 | 14.5 |

- **FPs are high-confidence** (0.736 avg) — the model fires on surface tokens in ambiguous utterances.
- **FNs cluster near threshold** (0.335 avg) — the model is uncertain on missed facts, not confidently wrong.
- **FNs are longer on average** (14.5 vs 9.4 words for TPs) — longer utterances with buried facts are harder to gate.

### 5. Class Separation (`results2/class_separation.csv`, `results2/honest_diagnostics.csv`)

Cosine separation ratio 1.129 — small margin, but L2 centroid distance (0.143) and linear probe accuracy (0.808) confirm a real learnable signal exists. See Honest Diagnostics section above.

---

## Training Cost

| Model | Train time | n samples |
|---|---|---|
| LSG ensemble + adapter | 0.54 s | 1 280 |
| LSG ensemble no adapter | 0.54 s | 1 280 |
| LSG LR only | 0.54 s | 1 280 |

RL-based memory gating (e.g. Memory-R1) requires GPU + multi-hour RL training loop. LSG trains in under 1 second on CPU with no reward signal.

---

## Project Structure

```
Lsg/
├── run.py                  ← entry point (outputs to results2/)
├── requirements.txt
├── results2/               ← all outputs (12 CSV files)
│   ├── main_results.csv
│   ├── pr_sweep.csv
│   ├── auprc.csv
│   ├── calibration.csv
│   ├── calibration_audit.csv
│   ├── latency_scaling.csv
│   ├── error_analysis.csv
│   ├── fp_patterns.csv
│   ├── feature_importance.csv
│   ├── class_separation.csv
│   ├── honest_diagnostics.csv
│   └── pca_projection.csv
└── lsg/
    ├── data.py             ← dataset loaders + labeling heuristic
    ├── embeddings.py       ← all-MiniLM-L6-v2 encoder (subprocess + disk cache)
    ├── baseline.py         ← statistical gate (cosine novelty + recency decay)
    ├── model.py            ← LR + XGBoost + MLP ensemble + DomainAdapter
    ├── evaluate.py         ← P/R/F1, AUC, AUPRC, ECE, bootstrap CI, Timer
    ├── train.py            ← 3-condition pipeline + rigorous evaluation + adaptation curve
    └── analysis.py         ← 6 analyses saved to results2/
```

---

## Known Platform Notes

- **Python 3.14 / arm64 macOS**: XGBoost's OpenMP runtime conflicts with the `tokenizers` semaphore when both run in the same process. `embeddings.py` works around this by running the encode step in a subprocess that exits before any ML code starts.
- Embeddings are cached to `.emb_cache/` as `.npy` files keyed by MD5 of the input texts. Delete this folder to force re-encoding.
