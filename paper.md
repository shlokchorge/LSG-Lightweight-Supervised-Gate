# LSG: A Lightweight Supervised Gate for Agentic LLM Memory
**Trading Threshold Heuristics for a Shallow, Calibrated, Domain-Adaptive Classifier**

**Shlok Chorge**  
*Department of Information Technology*  
*Vidyalankar Polytechnic and Padmabhushan Vasantdada Patil College of Engineering (VPPCOE)*  
*University of Mumbai, Mumbai, India*  
*Email: shlokchorge2929@gmail.com*  

**Prithibe Majumder**  
*Email: majumderprithibe@gmail.com*  

---

## Abstract
Deployed LLM agents increasingly rely on persistent memory stores to accumulate facts across long-running interactions. However, existing architectures gate memory writes using a fixed cosine similarity threshold—a rule that is computationally $O(n)$ with memory size $n$, domain-blind, and uncalibrated. We present **LSG (Lightweight Supervised Gate)**, a shallow supervised ensemble combining Logistic Regression, XGBoost, and a Multi-Layer Perceptron over frozen 384-dimensional Sentence-BERT (`all-MiniLM-L6-v2`) embeddings, augmented with a learned per-domain linear bias adapter (`DomainAdapter`). We reformulate memory gating into a binary classification task ($\text{STORE}=1$ vs. $\text{IGNORE}=0$) using heuristic labeling over Bitext customer support and PersonaChat dialogue datasets. In zero-shot cross-domain evaluation, LSG raises AUPRC from $0.468$ to $0.592$ ($+26.4\%$ relative improvement) and lowers Expected Calibration Error from $0.233$ to $0.187$ ($-19.8\%$ relative error reduction) compared to a cosine-similarity-plus-recency baseline. Furthermore, LSG achieves constant-time $O(1)$ per-decision inference latency ($0.777\text{ ms}$), providing up to an $11.1\times$ speedup at $n=1,000$ items. We also report empirical in-domain ceiling performance ($F_1 = 0.761$), few-shot adaptation curves ($k \in \{5, 20, 100\}$), multi-seed variance ($\pm 0.008$ AUPRC), feature importances, and false-positive error patterns.

**Index Terms**—agentic memory, novelty gating, LLM agents, calibration, domain adaptation, lightweight classifiers, ensemble learning.

---

## I. INTRODUCTION

Long-running Large Language Model (LLM) agents accumulate information across conversational sessions—including user preferences, task progress, named entities, and environmental constraints—within an external memory store that is retrieved into context on demand [1]–[3]. The practical operational efficiency and retrieval accuracy of an agent's memory depend fundamentally on decisions made at write time. For every incoming user or system utterance, the gate must decide whether to commit the text to durable long-term storage ($\text{STORE}$) or discard it as transient filler ($\text{IGNORE}$). Storing indiscriminately leads to severe memory store bloat, increasing retrieval latency and diluting the LLM's context window with uninformative conversational noise (e.g., *"okay"*, *"sounds good"*, *"tell me more"*). Conversely, discarding too aggressively causes catastrophic forgetting of critical user facts needed in future turns.

```
Incoming Utterance x ──► Sentence Embedding (all-MiniLM-L6-v2) ──► 384-d Vector
                                                                         │
                                                                         ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                         LSG Lightweight Supervised Gate                        │
│                                                                                │
│   ┌─────────────────────┐   ┌─────────────────────┐   ┌────────────────────┐   │
│   │ Logistic Regression │   │  XGBoost Classifier │   │   MLP (256 ──► 64) │   │
│   │    (Weight: 0.25)   │   │    (Weight: 0.50)   │   │   (Weight: 0.25)   │   │
│   └──────────┬──────────┘   └──────────┬──────────┘   └─────────┬──────────┘   │
│              └─────────────────────────┼────────────────────────┘              │
│                                        ▼                                       │
│                         Weighted Soft Voting Probability                       │
│                                        │                                       │
│                                        ▼                                       │
│                     DomainAdapter (Bias & Rank-1 Correction)                   │
│                                        │                                       │
│                                        ▼                                       │
│                      Threshold Calibration (Few-Shot Target)                   │
└────────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                                         ▼
                             Decision y ∈ {STORE, IGNORE}
```
*Figure 1. Architectural schematic of the LSG pipeline, showing frozen feature extraction, weighted tri-model ensemble classification, per-domain linear adaptation, and few-shot threshold calibration.*

In practice, memory write gating is overwhelmingly implemented using fixed-threshold statistical rules: the incoming utterance is embedded into a dense vector space, its cosine similarity to existing stored memory vectors is computed, optionally discounted by recency decay, and stored if the combined novelty score exceeds a hand-tuned cutoff. While conceptually straightforward and training-free, fixed similarity heuristics suffer from three structural weaknesses that motivate this work:

1. **No Notion of Content Type:** A similarity-only gate evaluates semantic distance without understanding informational value. A novel piece of generic small talk (e.g., *"I love rainy Tuesdays"*) and a novel personal fact (e.g., *"I am allergic to peanuts"*) appear equally distant from existing memory vectors; both are incorrectly flagged as storeworthy "novelty."
2. **Domain Blindness:** A single global similarity threshold cannot dynamically adapt across domains where the baseline rate of storeworthy content differs drastically (e.g., entity-dense task-oriented customer support vs. casual open-domain chit-chat).
3. **Poor Calibration and Unbounded $O(n)$ Cost:** The confidence score returned by a distance-based similarity gate is uncalibrated and does not correspond to a posterior probability. Furthermore, evaluating every new turn requires scanning against all $n$ existing memory vectors, resulting in $O(n)$ time complexity that degrades linearly as memory expands.

To solve all three structural problems simultaneously without requiring expensive GPU training or complex Reinforcement Learning (RL) reward modeling, we introduce **LSG (Lightweight Supervised Gate)**. LSG leverages frozen, CPU-friendly sentence transformer embeddings (`all-MiniLM-L6-v2`), a tri-model CPU-trainable ensemble (Logistic Regression, XGBoost, and MLP), and a parameter-efficient per-domain linear bias shift adapter (`DomainAdapter`). Trained end-to-end in seconds on standard CPU hardware, LSG reformulates memory gating into a calibrated, selective classification task.

### Summary of Key Contributions
- **Supervised Gating Formulation:** We formalize agent memory write gating as a binary classification task ($\text{STORE}=1$ vs. $\text{IGNORE}=0$) over dual dialogue domains: Bitext customer support (task-oriented) and PersonaChat (open-domain chat).
- **Lightweight CPU Architecture:** We design LSG, combining frozen SBERT embeddings with a weighted Logistic Regression + XGBoost + MLP ensemble and a per-domain linear adapter (`DomainAdapter`), trainable end-to-end on a laptop CPU in seconds.
- **Constant-Time $O(1)$ Inference:** We achieve flat per-decision inference latency ($0.777\text{ ms}$), providing an $11.1\times$ speedup over the statistical baseline's $O(n)$ scan at $n=1,000$ items.
- **Empirical Superiority in Ranking and Calibration:** We demonstrate that zero-shot LSG significantly outperforms the similarity baseline in ranking quality (AUPRC: $0.592$ vs. $0.468$, $+26.4\%$ relative improvement) and Expected Calibration Error (ECE: $0.187$ vs. $0.233$, $-19.8\%$ relative error reduction).
- **Comprehensive Diagnostic Audit:** We report multi-seed variance ($\pm 0.008$ AUPRC), 95% bootstrap confidence intervals, bidirectional adaptation curves ($k \in \{5, 20, 100\}$), feature importance rankings, embedding geometry metrics, and an empirical breakdown of false-positive error patterns.

---

## II. RELATED WORK

Agent memory architectures have seen substantial development in recent years. MemGPT (Letta) [1] models an LLM's context window as an operating system memory hierarchy, managing page swaps between fast working memory and long-term storage. Mem0 [2] provides production-grade key-value memory extraction and retrieval via dense vector indexes. A-MEM [3] constructs a self-organizing memory graph utilizing Zettelkasten-style dynamic linking and entry superseding. However, these systems focus primarily on downstream memory retrieval, page organization, and link management. None treat the initial write-time store/ignore decision itself as a calibrated, supervised classification problem with explicit latency constraints—the exact research gap targeted by LSG.

Sentence transformer models, such as Sentence-BERT [4], generate low-dimensional dense vector representations where spatial distance reflects semantic similarity. Using frozen sentence embeddings as input features for lightweight downstream classifiers is a proven approach that minimizes computational overhead while maintaining high classification accuracy across diverse natural language processing tasks. XGBoost [5] offers fast, non-linear tree-based ensemble learning, while scikit-learn [6] provides standardized, highly optimized CPU implementations for Logistic Regression and Multi-Layer Perceptrons.

Model calibration measures the degree to which a classifier's predicted probability scores align with empirical accuracy. Expected Calibration Error (ECE) [7] quantifies this alignment using equal-width confidence binning. Proper calibration is vital in autonomous agent systems because downstream decision-making modules rely directly on gate confidence to decide whether to trigger memory storage pipelines.

We evaluate LSG across two contrasting dialogue corpora representing distinct conversational domains. Bitext Customer Support [8] provides slot- and entity-rich task-oriented user instructions and chatbot responses. PersonaChat [9] supplies persona-grounded chit-chat containing explicit first-person factual statements along with general conversation. Treating these corpora as separate domains enables rigorous evaluation of cross-domain transfer and few-shot domain adaptation.

---

## III. PROBLEM FORMULATION

Let $\mathcal{U} = \{x_1, x_2, \dots, x_N\}$ denote a streaming sequence of conversational turns within an agent-user session. For each incoming utterance $x_i$, a memory gate $g_\theta$ must output a binary decision $y_i \in \{0, 1\}$:
$$y_i = \begin{cases} 1 & (\text{STORE: durable fact, preference, entity, or slot value}) \\ 0 & (\text{IGNORE: transient filler, backchannel, generic question}) \end{cases}$$

The gate function $g_\theta(x_i) \mapsto \hat{p}_i \in [0, 1]$ estimates the posterior probability $P(y_i = 1 \mid x_i)$. Applying a decision threshold $\tau \in (0, 1)$ yields the final discrete gating output:
$$\hat{y}_i = \mathbb{I}(\hat{p}_i \ge \tau)$$

We evaluate $g_\theta$ across three operational dimensions:
1. **Ranking Quality:** Threshold-independent metrics including Area Under the Receiver Operating Characteristic curve ($\text{ROC-AUC}$) and Area Under the Precision-Recall Curve ($\text{AUPRC}$).
2. **Decision Quality:** Precision ($P$), Recall ($R$), and $F_1$-score evaluated at threshold $\tau = 0.5$ and at calibrated optimal thresholds.
3. **Calibration & Efficiency:** Expected Calibration Error ($\text{ECE}$) and per-decision wall-clock inference latency $t_{\text{inf}}$ as existing memory store size $n$ grows from $10$ to $1,000$ items.

---

## IV. METHODOLOGY

### A. Automatic Heuristic Labeling
To generate training labels without manual annotation cost, we construct an automated rule-based labeling function $L(x) \mapsto y \in \{0, 1\}$ targeting distinct informativeness patterns:

- **Persona Statements ($\text{STORE}=1$):** Regular expression matching first-person factual assertions:
  $$\text{regex: } \text{\textasciicircum i (am|'m|have|like|love|hate|prefer|work|live|own|play|enjoy|want|need|studied|grew up)}\b$$
- **Task Entities and Slots ($\text{STORE}=1$):** Regular expressions matching domain slots (e.g., `{{slot}}`), named entities, and key transaction terms (*order, refund, delivery, payment, reservation, check-in, account*).
- **Backchannel and Filler ($\text{IGNORE}=0$):** Regular expressions catching generic conversational turns:
  $$\text{regex: } \text{\textasciicircum (ok|okay|yes|no|sure|thanks|thank you|hello|hi|bye|got it|sounds good|nice|cool|what is|do you)}[\dots]*\$$$
- **Length Constraint:** Utterances with $\le 5$ words that fail to match any entity or preference regex are automatically designated as $\text{IGNORE}=0$.

```python
# lsg/data.py - Labeling Logic Excerpt
def _label(text: str) -> int:
    text = text.strip()
    if _IGNORE_RE.match(text):
        return 0
    if _PERSONA_RE.match(text) or _SLOT_RE.search(text) or _STORE_RE.search(text) or _NER_RE.search(text):
        return 1
    if len(text.split()) <= 5:
        return 0
    return 0
```

### B. Feature Extraction and Subprocess Caching
All input text turns are converted into 384-dimensional dense vectors using the frozen `all-MiniLM-L6-v2` Sentence-BERT model [4]:
$$\mathbf{e}_i = \text{SBERT}(x_i) \in \mathbb{R}^{384}$$

To prevent process deadlock and OpenMP runtime conflicts between `sentence-transformers` and `XGBoost` on multi-threading platforms, feature extraction is isolated inside a dedicated Python subprocess. Computed embeddings are hashed via MD5 and stored to disk in `.emb_cache/` for instant retrieval across experiment runs.

### C. Statistical Novelty Baseline
The statistical baseline simulates streaming memory without parameter updates. Given incoming embedding $\mathbf{e}_i$ and stored memory matrix $\mathbf{M}_{<i} = [\mathbf{m}_1, \dots, \mathbf{m}_{n}]^T$, it computes recency-weighted maximum cosine similarity:
$$s_{\text{sim}}(\mathbf{e}_i, \mathbf{M}_{<i}) = \max_{1 \le j \le n} \left( \frac{\mathbf{e}_i \cdot \mathbf{m}_j}{\|\mathbf{e}_i\| \|\mathbf{m}_j\|} \times \gamma^{i - j} \right)$$
where $\gamma = 0.95$ is the recency decay factor per turn. The baseline assigns novelty probability $\hat{p}_i = 1.0 - s_{\text{sim}}$ and stores $\mathbf{e}_i$ if $s_{\text{sim}} < \text{novelty\_thresh} = 0.85$. This baseline requires $O(n)$ operations per turn.

### D. LSG Ensemble Architecture
LSG combines three complementary CPU-trainable classifiers over frozen 384-dimensional embeddings:

1. **Logistic Regression ($\mathbf{m}_{\text{LR}}$):** $L_2$-regularized linear classifier ($C=0.5$, `max_iter=2000`, `class_weight='balanced'`).
2. **XGBoost Classifier ($\mathbf{m}_{\text{XGB}}$):** Gradient-boosted decision trees (`n_estimators=300`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`).
3. **Multi-Layer Perceptron ($\mathbf{m}_{\text{MLP}}$):** Neural network with architecture $384 \to 256 \to 64 \to 1$ (`max_iter=400`, `alpha=1e-3`, `early_stopping=True`).

The raw ensemble output is computed via weighted probability averaging:
$$\hat{p}_{\text{raw}}(\mathbf{x}) = w_{\text{LR}} \cdot P_{\text{LR}}(1 \mid \mathbf{x}) + w_{\text{XGB}} \cdot P_{\text{XGB}}(1 \mid \mathbf{x}) + w_{\text{MLP}} \cdot P_{\text{MLP}}(1 \mid \mathbf{x})$$
with fixed weights $\mathbf{w} = (0.25, 0.50, 0.25)$.

### E. DomainAdapter and Few-Shot Threshold Calibration
To handle cross-domain distribution shifts without retraining ensemble base models, LSG incorporates a linear `DomainAdapter`:
$$\mathbf{x}_{\text{adapted}} = \mathbf{x} \mathbf{W}_d^T + \mathbf{b}_d$$
where $\mathbf{b}_d \in \mathbb{R}^{384}$ is initialized to the domain mean displacement $(\bar{\mathbf{X}}_{\text{global}} - \bar{\mathbf{X}}_d)$ and $\mathbf{W}_d \in \mathbb{R}^{384 \times 384}$ is initialized to the identity matrix $\mathbf{I}_{384}$. When provided with a small budget of $k$ target-domain labeled examples $(k \in \{5, 20, 100\})$, `DomainAdapter.adapt()` performs gradient descent optimization over $\mathbf{b}_d$ and rank-1 updates on $\mathbf{W}_d$ to maximize class centroid margins:

$$\min_{\mathbf{b}_d, \mathbf{W}_d} \left\| \bar{\mathbf{x}}_{\text{adapted}}^{(1)} - \bar{\mathbf{x}}_{\text{adapted}}^{(0)} \right\|_2^{-1}$$

Following domain adaptation, LSG recalibrates its global decision threshold $\tau$ on the few-shot target set by sweeping precision-recall curves to maximize target $F_1$-score:
$$\tau^* = \arg\max_{\tau} \left( \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau) + \epsilon} \right)$$

---

## V. EXPERIMENTAL SETUP

### A. Datasets and Domain Splits
We load $2,000$ samples from Bitext customer support (Task-Oriented domain) and $2,000$ samples from PersonaChat (Open-Domain chat), yielding $4,000$ total turns. 

Table I outlines the overall dataset statistics and class distribution across domains.

| Domain | Dataset Source | Total Samples | STORE ($y=1$) | IGNORE ($y=0$) | Class Balance (% STORE) |
|---|---|---|---|---|---|
| **Task-Oriented** | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | 2,000 | 1,000 | 1,000 | 50.0% |
| **Open-Domain** | `AlekseyKorshuk/persona-chat` | 2,000 | 1,073 | 927 | 53.65% |
| **Combined Total** | Mixed Dialogue Corpus | 4,000 | 2,073 | 1,927 | 51.8% |

*Table I. Dataset composition, domain sources, and ground-truth label distributions.*

Experimental splits are generated using stratified sampling (`seed=42`):
- **Source Domain Train Set (`src_tr`):** 1,600 Bitext samples (80%).
- **Source Domain Validation Set (`src_val`):** 400 Bitext samples (20%).
- **Target Few-Shot Adaptation Pool (`pc_pool`):** 200 PersonaChat samples.
- **Target Evaluation Test Set (`pc_te`):** 1,800 PersonaChat samples.
- **In-Domain Benchmark Splits (`pc_id_tr` / `pc_id_te`):** 1,200 train / 800 test PersonaChat samples for ceiling comparison.

### B. Evaluation Metrics
Models are evaluated across five metrics:
1. **Precision ($P$), Recall ($R$), $F_1$-Score:** Evaluated on the positive $\text{STORE}$ class ($y=1$).
2. **ROC-AUC:** Area under the Receiver Operating Characteristic curve.
3. **AUPRC:** Area under the Precision-Recall Curve (primary threshold-free metric).
4. **Expected Calibration Error (ECE):** Formulated with $M=10$ equal-width bins:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
5. **Per-Decision Latency ($t_{\text{inf}}$):** Average inference wall-clock time per sample in milliseconds, benchmarked across memory capacities $n \in \{10, 50, 100, 200, 500, 1000\}$.

---

## VI. RESULTS AND EMPIRICAL FINDINGS

### A. Main Cross-Domain Comparison
Table II summarizes the primary zero-shot cross-domain performance (trained on Bitext, tested on PersonaChat) against the baseline and in-domain ceiling.

| Condition / Model | Precision | Recall | $F_1$-Score | ROC-AUC | AUPRC | ECE | Train Time (s) | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Statistical Baseline** | 0.5347 | **1.0000** | 0.6968 | 0.4149 | 0.4680 | 0.2331 | **0.00** | 8.6137 |
| **LSG Zero-Shot (No Adapter)** | 0.5548 | 0.8160 | 0.6605 | 0.5830 | 0.5917 | 0.1870 | 4.55 | **0.7774** |
| **LSG Few-Shot ($N=100$)** | 0.5961 | 0.8380 | 0.6964 | 0.5689 | 0.6040 | 0.2594 | 4.71 | **0.7774** |
| **LSG In-Domain Ceiling** | **0.6582** | 0.9023 | **0.7611** | **0.8142** | **0.8120** | **0.0350** | 4.31 | **0.7774** |

*Table II. Main evaluation results comparing the Statistical Baseline, LSG Zero-Shot, LSG Few-Shot ($N=100$), and LSG In-Domain Ceiling on PersonaChat test target data.*

```
                 Precision-Recall Curve Comparison
  1.0 ┌──────────────────────────────────────────────────────────┐
      │                                                          │
  0.8 │                                                          │
P     │    ────── LSG Ensemble (AUPRC = 0.592)                   │
r 0.6 │    - - -  Baseline Gating (AUPRC = 0.468)                 │
e     │  ──────────────────────────────                          │
c 0.4 │                                ──────────────            │
i     │                                              ─────────── │
o 0.2 │                                                          │
n     │                                                          │
  0.0 └──────────────────────────────────────────────────────────┘
      0.0        0.2        0.4        0.6        0.8        1.0
                               Recall
```
*Figure 2. Precision-Recall curves comparing LSG against the Statistical Baseline across the full decision threshold spectrum.*

#### Analysis of Main Results
- **Ranking and Calibration Advantage:** LSG outperforms the baseline on threshold-free metrics, boosting AUPRC from $0.4680$ to $0.5917$ ($+26.4\%$ relative improvement) and ROC-AUC from $0.4149$ to $0.5830$. Calibration error decreases from $0.2331$ to $0.1870$ ($-19.8\%$ relative drop).
- **The $F_1$ Paradox:** At the default threshold $\tau=0.5$, the baseline records a superficially higher $F_1$ ($0.6968$ vs. $0.6605$) solely because its recall is $1.0000$ (storing 100% of incoming items). Its precision ($0.5347$) is barely above the dataset base rate ($53.65\%$). The baseline acts as a pass-through filter rather than a selective gate. LSG achieves real selectivity, trading raw pass-through recall for precision and calibration.

### B. Few-Shot Adaptation Curve
We evaluate how LSG adapts to target domain shifts as the budget of target-domain labeled examples $N$ increases from $0$ to $200$.

| Adaptation Budget $N$ | $F_1$-Score | ROC-AUC | ECE | Operational Notes |
|---|---|---|---|---|
| **$N = 0$ (Zero-Shot)** | 0.6605 | 0.5830 | 0.1870 | Out-of-the-box cross-domain transfer |
| **$N = 10$** | 0.6689 | 0.5516 | 0.4233 | Initial adapter shift |
| **$N = 25$** | 0.6919 | 0.5509 | 0.4471 | Threshold shifts upward |
| **$N = 50$** | 0.6887 | 0.5584 | **0.1383** | Calibration optimal point |
| **$N = 100$** | 0.6964 | 0.5689 | 0.2594 | Matches baseline $F_1$ with high selectivity |
| **$N = \text{Ceiling}$** | **0.7611** | **0.8142** | **0.0350** | Full target in-domain supervision |

*Table III. Few-shot adaptation curve on PersonaChat test set across varying example budgets $N$.*

### C. Multi-Seed Variance and Ablation Studies
To isolate the contribution of each algorithmic component, we perform multi-seed evaluations ($N_{\text{seeds}} = 5$) and single-model ablations on the Bitext $\to$ PersonaChat transfer task.

| Model Variant | $F_1$ (Mean $\pm$ Std) | AUPRC (Mean $\pm$ Std) | ROC-AUC (Mean $\pm$ Std) | Training Time (s) |
|---|---|---|---|---|
| **LSG Full Ensemble + Adapter** | **0.660 $\pm$ 0.005** | **0.583 $\pm$ 0.008** | **0.580 $\pm$ 0.007** | 4.31 |
| **LSG Ensemble (No Adapter)** | 0.660 $\pm$ 0.004 | 0.583 $\pm$ 0.007 | 0.580 $\pm$ 0.006 | 4.55 |
| **Logistic Regression Only** | 0.657 $\pm$ 0.004 | 0.570 $\pm$ 0.004 | 0.556 $\pm$ 0.002 | **0.10** |
| **XGBoost Only** | 0.642 $\pm$ 0.006 | 0.561 $\pm$ 0.005 | 0.548 $\pm$ 0.005 | 2.15 |
| **MLP Classifier Only** | 0.648 $\pm$ 0.008 | 0.565 $\pm$ 0.009 | 0.559 $\pm$ 0.008 | 1.95 |
| **TF-IDF Vectorizer + Ensemble** | 0.640 $\pm$ 0.000 | 0.536 $\pm$ 0.000 | 0.538 $\pm$ 0.000 | **0.10** |

*Table IV. Ablation comparison showing multi-seed mean and standard deviation across ensemble configurations and feature representations.*

Non-parametric bootstrap resampling ($1,000$ iterations, $\alpha=0.05$) yields the following 95% confidence intervals for the full LSG model:
- **AUPRC 95% CI:** $[0.551, 0.613]$
- **$F_1$-Score 95% CI:** $[0.638, 0.682]$

### D. Bidirectional Domain Adaptation
We evaluate bidirectionality by swapping source and target domains (Bitext $\to$ PersonaChat vs. PersonaChat $\to$ Bitext).

| Direction | Few-Shot $k$ | $F_1$-Score | $\Delta F_1$ | AUPRC | $\Delta \text{AUPRC}$ |
|---|---|---|---|---|---|
| **Bitext $\to$ PersonaChat** | $k=0$ (Zero-Shot) | 0.667 | — | 0.579 | — |
| | $k=5$ | 0.611 | $-0.056$ | 0.577 | $-0.002$ |
| | $k=20$ | 0.693 | $+0.026$ | 0.600 | $+0.021$ |
| | $k=100$ | **0.695** | $+0.028$ | **0.604** | $+0.025$ |
| **PersonaChat $\to$ Bitext** | $k=0$ (Zero-Shot) | 0.585 | — | 0.632 | — |
| | $k=5$ | 0.585 | $+0.000$ | 0.632 | $+0.000$ |
| | $k=20$ | 0.611 | $+0.026$ | 0.655 | $+0.023$ |
| | $k=100$ | **0.611** | $+0.026$ | **0.656** | $+0.024$ |

*Table V. Bidirectional domain adaptation metrics across varying few-shot budgets $k$.*

### E. Per-Decision Latency and Memory Scaling
We measure wall-clock inference latency per decision as existing memory store size $n$ increases.

| Memory Size ($n$) | LSG Latency (ms) | Baseline Latency (ms) | Speedup Ratio ($\times$) | Complexity Class |
|---|---|---|---|---|
| **$n = 10$** | **0.7774** | 1.5444 | $2.0\times$ | $O(1)$ vs. $O(n)$ |
| **$n = 50$** | **0.7774** | 1.9656 | $2.5\times$ | $O(1)$ vs. $O(n)$ |
| **$n = 100$** | **0.7774** | 1.8278 | $2.4\times$ | $O(1)$ vs. $O(n)$ |
| **$n = 200$** | **0.7774** | 3.4017 | $4.4\times$ | $O(1)$ vs. $O(n)$ |
| **$n = 500$** | **0.7774** | 3.6578 | $4.7\times$ | $O(1)$ vs. $O(n)$ |
| **$n = 1,000$** | **0.7774** | 8.6137 | **$11.1\times$** | $O(1)$ vs. $O(n)$ |

*Table VI. Per-decision inference latency (ms) comparison across memory store capacities $n$.*

```
                       Inference Latency vs. Memory Size
  10.0 ┌─────────────────────────────────────────────────────────┐
       │                                                      ▲  │
   8.0 │                                                     /   │  Baseline O(n)
M  6.0 │                                                    /    │
s      │                                                   /     │
   4.0 │                                       ▲──────────┘      │
   2.0 │                           ▲───────────┘                 │
       │  ■────────────────────────┴───────────────────────────■ │  LSG O(1)
   0.0 └─────────────────────────────────────────────────────────┘
       10        100       200        500                   1000
                            Memory Size (n)
```
*Figure 3. Latency comparison showing constant-time $O(1)$ inference for LSG ($0.777\text{ ms}$) versus linear $O(n)$ time growth for the similarity baseline ($8.614\text{ ms}$ at $n=1000$).*

### F. Feature Importance and Embedding Space Diagnostics
To understand how LSG extracts decisions from frozen SBERT vectors, we inspect feature weights and geometric metrics.

| Metric / Dimension | Value | Analytical Interpretation |
|---|---|---|
| **Within-STORE Cosine Distance** | 0.1654 | High baseline similarity within positive class |
| **Within-IGNORE Cosine Distance** | 0.1465 | High baseline similarity within negative class |
| **Across-Class Cosine Distance** | 0.1382 | Classes overlap significantly in raw vector space |
| **Cosine Separation Ratio** | **1.1286** | Weak geometric separation ($>1.0$ threshold) |
| **$L_2$ Centroid Distance** | 0.1430 | Absolute Euclidean displacement between class centroids |
| **LDA Fisher Ratio** | 0.0115 | Small non-zero linear class separation signal |
| **Linear Probe Test Accuracy** | **80.78%** | Supervised upper-bound separability on target test set |

*Table VII. Embedding-space geometric metrics and linear probe upper bound.*

```
                 Top-10 Logistic Regression Feature Importances
    Dim 238 ────────────────────────────────────────────── 2.029
    Dim 301 ─────────────────────────────────────── 1.711
    Dim 330 ─────────────────────────────────── 1.545
    Dim 84  ────────────────────────────────── 1.495
    Dim 62  ────────────────────────────────── 1.490
    Dim 137 ─────────────────────────────── 1.436
    Dim 80  ───────────────────────────── 1.364
    Dim 14  ──────────────────────────── 1.353
    Dim 144 ──────────────────────────── 1.344
    Dim 17  ──────────────────────────── 1.340
            0.0         0.5         1.0         1.5         2.0
                                |Coefficient|
```
*Figure 4. Absolute magnitude of top-10 Logistic Regression feature coefficients across the 384 embedding dimensions.*

Figure 4 demonstrates that predictive weight is distributed broadly across multiple dimensions (top coefficient $2.029$ at dimension $238$, 10th-ranked $1.340$ at dimension $17$). This indicates that storeworthiness is a distributed semantic feature rather than an axis-aligned property.

### G. Error Audit and Calibration Breakdown
Analysis of the zero-shot test set ($N=1,800$) yields the confusion breakdown illustrated in Figure 5:
- **True Positives (TP):** 793 turns
- **False Positives (FP):** 640 turns
- **True Negatives (TN):** 198 turns
- **False Negatives (FN):** 169 turns

```
                       LSG Zero-Shot Error Breakdown
  800 ┌──────────────────────────────────────────────────────────┐
      │  ┌───────┐                                               │
  600 │  │  793  │           ┌───────┐                           │
C     │  │       │           │  640  │                           │
o 400 │  │       │           │       │                           │
u     │  │       │           │       │   ┌───────┐   ┌───────┐   │
n 200 │  │       │           │       │   │  198  │   │  169  │   │
t     │  │       │           │       │   │       │   │       │   │
    0 └─┴───────┴───────────┴───────┴───┴───────┴───┴───────┴───┘
            TP                  FP          TN          FN
```
*Figure 5. Distribution of predictions on the PersonaChat zero-shot test set ($N=1,800$).*

Categorization of the 640 False Positive errors reveals clear qualitative patterns:
1. **`longer_ambiguous` (493 cases, avg. confidence 0.726):** Social pleasantries or general chat utterances containing incidental entity-like words or locations (e.g., *"I was thinking about visiting Paris someday"*).
2. **`short_question` (108 cases, avg. confidence 0.748):** Conversational questions containing surface nouns (e.g., *"What is your favorite book?"*).
3. **`short_other` (29 cases, avg. confidence 0.763):** Short social assertions with ambiguous storeworthiness.

Detailed bin-by-bin calibration analysis (Table VIII) shows that while LSG reduces ECE relative to the baseline ($0.1870$ vs. $0.2331$), it exhibits overconfidence in the mid-to-high probability range ($0.4 \le \hat{p} \le 1.0$).

| Confidence Bin | Sample Count $n$ | Mean Confidence | Actual Accuracy | Signed Gap | Calibration Status |
|---|---|---|---|---|---|
| **$[0.0, 0.1)$** | 3 | 0.0866 | 0.3333 | $-0.2468$ | Underconfident |
| **$[0.1, 0.2)$** | 46 | 0.1562 | 0.4130 | $-0.2568$ | Underconfident |
| **$[0.2, 0.3)$** | 83 | 0.2573 | 0.4699 | $-0.2126$ | Underconfident |
| **$[0.3, 0.4)$** | 109 | 0.3495 | 0.5505 | $-0.2010$ | Underconfident |
| **$[0.4, 0.5)$** | 144 | 0.4512 | 0.4028 | $+0.0485$ | Overconfident (Non-monotonic) |
| **$[0.5, 0.6)$** | 183 | 0.5536 | 0.3934 | $+0.1602$ | Overconfident |
| **$[0.6, 0.7)$** | 262 | 0.6524 | 0.4809 | $+0.1715$ | Overconfident |
| **$[0.7, 0.8)$** | 406 | 0.7525 | 0.5961 | $+0.1564$ | Overconfident |
| **$[0.8, 0.9)$** | 465 | 0.8467 | 0.6065 | $+0.2403$ | Overconfident |
| **$[0.9, 1.0]$** | 99 | 0.9194 | 0.6364 | $+0.2831$ | Overconfident |

*Table VIII. Bin-by-bin calibration audit for LSG on the zero-shot test set ($N=1,800$).*

---

## VII. DISCUSSION AND LIMITATIONS

### Empirical Claims Supported by the Codebase
Our experimental pipeline empirically supports three core findings:
1. **Selective Gating vs. Blind Storage:** LSG eliminates the baseline's pass-through failure mode (storing 100% of items to maximize raw $F_1$), yielding a true selective gate with $+26.4\%$ higher AUPRC.
2. **Computational Scalability:** LSG replaces $O(n)$ vector comparisons with $O(1)$ forward passes, offering an $11.1\times$ speedup at $n=1,000$ memory items.
3. **Low Training Overhead:** The full tri-model ensemble trains in under $5\text{ seconds}$ on standard CPU hardware.

### Nuances and Limitations
- **Heuristic Label Noise:** Auto-generated regex labels serve as a surrogate for true human intent. Surface-level keyword matching mislabels certain social pleasantries as storeworthy, imposing an upper bound on supervised accuracy.
- **Mid-Range Overconfidence:** The calibration audit reveals that predicted probabilities between $0.4$ and $0.9$ systematically overestimate accuracy. Applying post-hoc temperature scaling or Platt scaling is a recommended direction for future deployment.

---

## VIII. CONCLUSION

We presented **LSG (Lightweight Supervised Gate)**, a shallow supervised ensemble classifier designed for agentic LLM memory gating. By combining Logistic Regression, XGBoost, and an MLP over frozen SBERT embeddings with a per-domain linear adapter, LSG addresses the key limitations of traditional similarity heuristics. Zero-shot cross-domain experiments demonstrate significant gains in ranking quality (AUPRC $0.592$ vs. $0.468$), lower calibration error (ECE $0.187$ vs. $0.233$), and constant-time $O(1)$ inference scalability ($0.777\text{ ms}$ per turn). LSG provides a practical, CPU-friendly foundation for building selective, fast, and reliable write-time memory systems in conversational AI agents.

---

## REFERENCES

[1] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez, "MemGPT: Towards LLMs as Operating Systems," *arXiv preprint arXiv:2310.08560*, 2023.  
[2] Mem0 Team, "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory," 2025.  
[3] W. Xu et al., "A-MEM: Agentic Memory for LLM Agents," *arXiv preprint arXiv:2502.12110*, 2025.  
[4] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP-IJCNLP*, 2019.  
[5] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. ACM SIGKDD*, 2016.  
[6] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.  
[7] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in *Proc. ICML*, 2017.  
[8] Bitext, "Bitext Customer Support LLM Chatbot Training Dataset," HuggingFace Datasets, `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, 2023.  
[9] S. Zhang, E. Dinan, J. Urbanek, A. Szlam, D. Kiela, and J. Weston, "Personalizing Dialogue Agents: I have a dog, do you have pets too?," in *Proc. ACL*, 2018.  
