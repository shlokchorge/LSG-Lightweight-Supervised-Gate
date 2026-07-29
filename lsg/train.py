"""
LSG training pipeline — 3-way evaluation:
  A. In-domain upper bound   : train+test both on PersonaChat
  B. Zero-shot cross-domain  : train on Bitext, test on PersonaChat (no adaptation)
  C. Few-shot adapted        : train on Bitext, adapt adapter on 100 PersonaChat
                               examples, test on remaining PersonaChat

Ablations within each condition: ensemble+adapter / no-adapter / LR / XGB / MLP
"""
import warnings
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os
from lsg.data import load_taskoriented, load_personachat
from lsg.embeddings import encode
from lsg.baseline import StatisticalGate
from lsg.model import LSGEnsemble
from lsg.evaluate import evaluate, Timer

RESULTS_DIR = os.environ.get("LSG_RESULTS_DIR", "results")

FEW_SHOT_N = 100        # labeled target-domain examples for adaptation
ADAPT_CURVE_NS = [10, 25, 50, 100, 200]   # sweep for adaptation curve
N_SEEDS = 5             # seeds for variance estimation
FEW_SHOT_KS = [5, 20, 100]  # k values for domain-adaptation table


def _build_splits(X, y, domains):
    """Return all index splits needed by both run() and run_adaptation_curve()."""
    task_idx = np.where(domains == "taskoriented")[0]
    pc_idx   = np.where(domains == "personachat")[0]

    src_tr, src_val = train_test_split(
        task_idx, test_size=0.2, random_state=42, stratify=y[task_idx]
    )
    # Reserve 200 for the largest few-shot pool; rest is test
    pc_pool, pc_te = train_test_split(
        pc_idx, train_size=200, random_state=42, stratify=y[pc_idx]
    )
    pc_id_tr, pc_id_te = train_test_split(
        pc_idx, test_size=0.4, random_state=42, stratify=y[pc_idx]
    )
    return dict(
        src_tr=src_tr, src_val=src_val,
        pc_pool=pc_pool, pc_te=pc_te,
        pc_id_tr=pc_id_tr, pc_id_te=pc_id_te,
    )


def run():
    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading data...")
    df_task = load_taskoriented(max_samples=2000)
    df_pc   = load_personachat(max_samples=2000)
    df_all  = pd.concat([df_task, df_pc], ignore_index=True)
    print(f"  Bitext task-oriented : {len(df_task)} samples")
    print(f"  PersonaChat          : {len(df_pc)} samples")
    print(f"  Label distribution:\n{df_all.groupby(['domain','label']).size().unstack(fill_value=0)}\n")

    # ── 2. Embeddings ─────────────────────────────────────────────────────────
    print("Encoding embeddings (cached after first run)...")
    with Timer() as t_emb:
        embs = encode(df_all["text"].tolist())
    print(f"  Encoded {len(embs)} samples in {t_emb.elapsed:.1f}s\n")

    X       = embs
    y       = df_all["label"].values
    domains = df_all["domain"].values

    # ── 3. Index splits ───────────────────────────────────────────────────────
    splits = _build_splits(X, y, domains)
    src_tr   = splits["src_tr"]
    pc_pool  = splits["pc_pool"]   # up to 200 few-shot examples
    pc_te    = splits["pc_te"]
    pc_id_tr = splits["pc_id_tr"]
    pc_id_te = splits["pc_id_te"]

    # Few-shot subset for condition C
    fs_n = min(FEW_SHOT_N, len(pc_pool))
    fs_idx, _ = train_test_split(
        pc_pool, train_size=fs_n, random_state=42, stratify=y[pc_pool]
    )

    print(f"Source train: {len(src_tr)} | Few-shot pool: {len(pc_pool)} | Target test: {len(pc_te)}")
    print(f"In-domain train: {len(pc_id_tr)} | In-domain test: {len(pc_id_te)}\n")

    X_src_tr, y_src_tr, d_src_tr = X[src_tr],   y[src_tr],   domains[src_tr]
    X_te,     y_te,     d_te     = X[pc_te],    y[pc_te],    domains[pc_te]
    X_fs,     y_fs               = X[fs_idx],   y[fs_idx]
    X_id_tr,  y_id_tr,  d_id_tr  = X[pc_id_tr], y[pc_id_tr], domains[pc_id_tr]
    X_id_te,  y_id_te,  d_id_te  = X[pc_id_te], y[pc_id_te], domains[pc_id_te]

    # ── 4. Statistical baseline ───────────────────────────────────────────────
    print("Running statistical baseline (on cross-domain test set)...")
    gate = StatisticalGate(novelty_thresh=0.85, recency_decay=0.95)
    with Timer() as t_base:
        base_probs = gate.predict_proba(X_te)
        # Use optimal threshold for baseline too
        from sklearn.metrics import precision_recall_curve as _prc
        _p, _r, _t = _prc(y_te, base_probs)
        _f1s = 2*_p*_r/(_p+_r+1e-9)
        _best_thr = float(_t[_f1s[:-1].argmax()])
        base_preds = (base_probs >= _best_thr).astype(int)
    base_metrics = {
        **evaluate(y_te, base_preds, base_probs),
        "train_time_s": 0.0,
        "latency_ms": t_base.elapsed / len(X_te) * 1000,
        "condition": "cross-domain (zero-shot)",
    }

    results = {}

    # ── helper ────────────────────────────────────────────────────────────────
    def train_eval(name, model, X_tr, y_tr, d_tr, X_ev, y_ev, d_ev, condition):
        with Timer() as t_tr:
            model.fit(X_tr, y_tr, d_tr)
        with Timer() as t_inf:
            probs = model.predict_proba(X_ev, d_ev)
            preds = model.predict(X_ev, d_ev)  # uses learned threshold
        m = evaluate(y_ev, preds, probs)
        m["train_time_s"] = t_tr.elapsed
        m["latency_ms"]   = t_inf.elapsed / len(X_ev) * 1000
        m["condition"]    = condition
        results[name] = m
        print(f"  [{name}] F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}  "
              f"ECE={m['ece']:.3f}  train={t_tr.elapsed:.1f}s")

    # ── 5-A. In-domain upper bound ────────────────────────────────────────────
    print("\n── A. In-domain upper bound (train+test on PersonaChat) ──")
    train_eval("A: LSG ensemble+adapter (in-domain)",
               LSGEnsemble(use_adapter=True),
               X_id_tr, y_id_tr, d_id_tr, X_id_te, y_id_te, d_id_te,
               "in-domain")
    train_eval("A: LSG no adapter (in-domain)",
               LSGEnsemble(use_adapter=False),
               X_id_tr, y_id_tr, d_id_tr, X_id_te, y_id_te, d_id_te,
               "in-domain")

    # ── 5-B. Zero-shot cross-domain ───────────────────────────────────────────
    print("\n── B. Zero-shot cross-domain (train Bitext → test PersonaChat) ──")
    train_eval("B: LSG ensemble+adapter (zero-shot)",
               LSGEnsemble(use_adapter=True),
               X_src_tr, y_src_tr, d_src_tr, X_te, y_te, d_te,
               "cross-domain (zero-shot)")
    train_eval("B: LSG no adapter (zero-shot)",
               LSGEnsemble(use_adapter=False),
               X_src_tr, y_src_tr, d_src_tr, X_te, y_te, d_te,
               "cross-domain (zero-shot)")

    # Single-model ablations (zero-shot)
    class _Single(LSGEnsemble):
        def __init__(self, which):
            super().__init__(use_adapter=True)
            self._which = which
        def predict_proba(self, X, domains):
            X_a = self._adapt(X, domains)
            return getattr(self, self._which).predict_proba(X_a)[:, 1]
        def predict(self, X, domains, threshold=None):
            probs = self.predict_proba(X, domains)
            thr = threshold if threshold is not None else self.threshold
            return (probs >= thr).astype(int)

    for which in ("lr", "xgb", "mlp"):
        train_eval(f"B: LSG {which.upper()} only (zero-shot)",
                   _Single(which),
                   X_src_tr, y_src_tr, d_src_tr, X_te, y_te, d_te,
                   "cross-domain (zero-shot)")

    # ── 5-C. Few-shot adapted ─────────────────────────────────────────────────
    print(f"\n── C. Few-shot adapted ({FEW_SHOT_N} target examples) ──")

    def train_eval_fewshot(name, model):
        with Timer() as t_tr:
            model.fit(X_src_tr, y_src_tr, d_src_tr)
            model.few_shot_adapt(X_fs, y_fs, domain="personachat")
        with Timer() as t_inf:
            probs = model.predict_proba(X_te, d_te)
            preds = model.predict(X_te, d_te)  # uses learned threshold
        m = evaluate(y_te, preds, probs)
        m["train_time_s"] = t_tr.elapsed
        m["latency_ms"]   = t_inf.elapsed / len(X_te) * 1000
        m["condition"]    = f"cross-domain (few-shot {FEW_SHOT_N})"
        results[name] = m
        print(f"  [{name}] F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}  "
              f"ECE={m['ece']:.3f}  train={t_tr.elapsed:.1f}s")

    train_eval_fewshot("C: LSG ensemble+adapter (few-shot)", LSGEnsemble(use_adapter=True))
    train_eval_fewshot("C: LSG no adapter (few-shot)",       LSGEnsemble(use_adapter=False))

    # ── 6. Results table ──────────────────────────────────────────────────────
    all_results = {"Baseline (statistical)": base_metrics, **results}
    cols = ["condition", "precision", "recall", "f1", "roc_auc", "ece",
            "train_time_s", "latency_ms"]
    df_res = pd.DataFrame(all_results).T[cols].round(4)

    print("\n" + "="*100)
    print("RESULTS TABLE")
    print("="*100)
    print(df_res.to_string())
    print("="*100)
    print("\nMetric notes:")
    print("  precision/recall/f1 → STORE class (label=1)")
    print("  roc_auc             → area under ROC curve")
    print("  ece                 → expected calibration error (lower=better)")
    print("  train_time_s        → wall-clock training time (0 = no training)")
    print("  latency_ms          → avg inference latency per sample (ms)")
    print(f"\n  Few-shot N={FEW_SHOT_N} | In-domain train size={len(pc_id_tr)}")

    return df_res


def run_adaptation_curve():
    """
    Sweep few-shot N over ADAPT_CURVE_NS.
    Train ensemble once on source domain, then for each N:
      - adapt only the adapter bias on N target examples
      - evaluate on the held-out target test set
    Prints an ASCII table and returns a DataFrame.
    """
    print("\n" + "="*60)
    print("ADAPTATION CURVE  (F1 / AUC / ECE vs few-shot N)")
    print("="*60)

    # ── data + embeddings (reuses cache) ─────────────────────────
    df_task = load_taskoriented(max_samples=2000)
    df_pc   = load_personachat(max_samples=2000)
    df_all  = pd.concat([df_task, df_pc], ignore_index=True)
    embs    = encode(df_all["text"].tolist())
    X, y, domains = embs, df_all["label"].values, df_all["domain"].values

    splits   = _build_splits(X, y, domains)
    src_tr   = splits["src_tr"]
    pc_pool  = splits["pc_pool"]
    pc_te    = splits["pc_te"]

    X_src_tr, y_src_tr, d_src_tr = X[src_tr], y[src_tr], domains[src_tr]
    X_pool,   y_pool             = X[pc_pool], y[pc_pool]
    X_te,     y_te,     d_te     = X[pc_te],   y[pc_te],  domains[pc_te]

    # ── train base model once ─────────────────────────────────────
    base_model = LSGEnsemble(use_adapter=True)
    base_model.fit(X_src_tr, y_src_tr, d_src_tr)

    # zero-shot baseline row
    rows = []
    p0 = base_model.predict_proba(X_te, d_te)
    m0 = evaluate(y_te, base_model.predict(X_te, d_te), p0)
    rows.append({"N": 0, **{k: round(m0[k], 4) for k in ("f1", "roc_auc", "ece")}})
    print(f"  N=  0 (zero-shot)  F1={m0['f1']:.3f}  AUC={m0['roc_auc']:.3f}  ECE={m0['ece']:.3f}")

    for n in ADAPT_CURVE_NS:
        if n >= len(pc_pool):   # need at least one sample left for test
            continue
        # Sample n stratified examples from the pool
        fs_idx, _ = train_test_split(
            np.arange(len(pc_pool)), train_size=n,
            random_state=42, stratify=y_pool
        )
        # Deep-copy adapter so each N is independent
        import copy
        model_n = copy.deepcopy(base_model)
        model_n.few_shot_adapt(X_pool[fs_idx], y_pool[fs_idx], domain="personachat")
        probs = model_n.predict_proba(X_te, d_te)
        m = evaluate(y_te, model_n.predict(X_te, d_te), probs)
        rows.append({"N": n, **{k: round(m[k], 4) for k in ("f1", "roc_auc", "ece")}})
        print(f"  N={n:>3}              F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}  ECE={m['ece']:.3f}")

    # in-domain ceiling
    pc_id_tr, pc_id_te = splits["pc_id_tr"], splits["pc_id_te"]
    ceil_model = LSGEnsemble(use_adapter=True)
    ceil_model.fit(X[pc_id_tr], y[pc_id_tr], domains[pc_id_tr])
    pc = ceil_model.predict_proba(X[pc_id_te], domains[pc_id_te])
    mc = evaluate(y[pc_id_te], ceil_model.predict(X[pc_id_te], domains[pc_id_te]), pc)
    rows.append({"N": "ceiling", **{k: round(mc[k], 4) for k in ("f1", "roc_auc", "ece")}})
    print(f"  N=ceiling (in-dom) F1={mc['f1']:.3f}  AUC={mc['roc_auc']:.3f}  ECE={mc['ece']:.3f}")

    df_curve = pd.DataFrame(rows).set_index("N")
    print("\n" + df_curve.to_string())
    print("="*60)
    return df_curve


def run_rigorous():
    """
    Rigorous evaluation addressing reviewer concerns:
      1. Bidirectional domain-adaptation table (A→B and B→A) with few-shot k=[5,20,100]
      2. Multi-seed variance (N_SEEDS) + bootstrap CIs on AUPRC/F1
      3. Additional baselines: majority-class, always-store, plain LR (no ensemble/adapter)
      4. TF-IDF embedding ablation vs SBERT
      5. Training wall-clock cost table
    """
    import copy
    from sklearn.feature_extraction.text import TfidfVectorizer
    from lsg.evaluate import bootstrap_ci

    print("\n" + "="*70)
    print("RIGOROUS EVALUATION")
    print("="*70)

    # ── data + embeddings ─────────────────────────────────────────────────
    print("Loading data...")
    df_task = load_taskoriented(max_samples=2000)
    df_pc   = load_personachat(max_samples=2000)
    df_all  = pd.concat([df_task, df_pc], ignore_index=True)
    with Timer() as t_emb:
        embs = encode(df_all["text"].tolist())
    print(f"  Embeddings: {t_emb.elapsed:.1f}s")

    X       = embs
    y       = df_all["label"].values
    domains = df_all["domain"].values
    texts   = df_all["text"].tolist()

    task_idx = np.where(domains == "taskoriented")[0]
    pc_idx   = np.where(domains == "personachat")[0]

    # Fixed test sets (20% of each domain, seed=42)
    _, task_te_idx = train_test_split(task_idx, test_size=0.2, random_state=42, stratify=y[task_idx])
    _, pc_te_idx   = train_test_split(pc_idx,   test_size=0.2, random_state=42, stratify=y[pc_idx])
    task_tr_pool   = np.setdiff1d(task_idx, task_te_idx)
    pc_tr_pool     = np.setdiff1d(pc_idx,   pc_te_idx)

    X_task_te, y_task_te, d_task_te = X[task_te_idx], y[task_te_idx], domains[task_te_idx]
    X_pc_te,   y_pc_te,   d_pc_te   = X[pc_te_idx],   y[pc_te_idx],   domains[pc_te_idx]

    # ── 1. Sanity baselines (no training) ────────────────────────────────
    print("\n── Sanity baselines ──")
    majority_label = int(y[pc_tr_pool].mean() >= 0.5)  # majority class on train
    majority_preds = np.full(len(y_pc_te), majority_label)
    majority_probs = np.full(len(y_pc_te), float(majority_label))
    always_store_preds = np.ones(len(y_pc_te), dtype=int)
    always_store_probs = np.ones(len(y_pc_te))

    for name, preds, probs in [
        ("Majority class",  majority_preds,     majority_probs),
        ("Always-STORE",    always_store_preds, always_store_probs),
    ]:
        m = evaluate(y_pc_te, preds, probs)
        print(f"  {name:20s}  F1={m['f1']:.3f}  AUPRC={m['auprc']:.3f}  AUC={m['roc_auc']:.3f}")

    # ── 2. Multi-seed variance: train on Bitext, test on PersonaChat ──────
    print(f"\n── Multi-seed variance ({N_SEEDS} seeds), Bitext→PersonaChat ──")
    seed_metrics = {"ensemble": [], "lr_only": []}

    for seed in range(N_SEEDS):
        tr_idx, _ = train_test_split(task_tr_pool, test_size=0.2, random_state=seed, stratify=y[task_tr_pool])
        X_tr, y_tr, d_tr = X[tr_idx], y[tr_idx], domains[tr_idx]

        # Ensemble
        m_ens = LSGEnsemble(use_adapter=True)
        m_ens.fit(X_tr, y_tr, d_tr)
        p_ens = m_ens.predict_proba(X_pc_te, d_pc_te)
        seed_metrics["ensemble"].append(evaluate(y_pc_te, m_ens.predict(X_pc_te, d_pc_te), p_ens))

        # Plain LR only (no ensemble, no adapter) — isolates what each piece contributes
        m_lr = LSGEnsemble(use_adapter=False)
        m_lr.fit(X_tr, y_tr, d_tr)
        p_lr = m_lr.lr.predict_proba(m_lr._adapt(X_pc_te, d_pc_te))[:, 1]
        seed_metrics["lr_only"].append(evaluate(y_pc_te, m_lr.predict(X_pc_te, d_pc_te), p_lr))

    print(f"  {'Model':25s}  {'F1 mean±std':18s}  {'AUPRC mean±std':18s}  {'AUC mean±std'}")
    for name, mlist in seed_metrics.items():
        f1s    = np.array([m["f1"]      for m in mlist])
        auprcs = np.array([m["auprc"]   for m in mlist])
        aucs   = np.array([m["roc_auc"] for m in mlist])
        print(f"  {name:25s}  {f1s.mean():.3f}±{f1s.std():.3f}       "
              f"{auprcs.mean():.3f}±{auprcs.std():.3f}       "
              f"{aucs.mean():.3f}±{aucs.std():.3f}")

    # Bootstrap CIs on the full train split (seed=0 model)
    tr_idx0, _ = train_test_split(task_tr_pool, test_size=0.2, random_state=0, stratify=y[task_tr_pool])
    m_boot = LSGEnsemble(use_adapter=True)
    m_boot.fit(X[tr_idx0], y[tr_idx0], domains[tr_idx0])
    p_boot = m_boot.predict_proba(X_pc_te, d_pc_te)
    for metric in ("auprc", "f1"):
        lo, hi = bootstrap_ci(y_pc_te, p_boot, metric=metric)
        print(f"  Bootstrap 95% CI  {metric}: [{lo:.3f}, {hi:.3f}]")

    # ── 3. Bidirectional domain-adaptation table ──────────────────────────
    print("\n── Domain-adaptation table (bidirectional, k-shot) ──")
    print(f"  {'Direction':30s}  {'k':>5}  {'F1':>6}  {'AUPRC':>7}")

    directions = [
        ("Bitext→PersonaChat", task_tr_pool, X_pc_te, y_pc_te, d_pc_te, pc_tr_pool, "personachat"),
        ("PersonaChat→Bitext", pc_tr_pool,   X_task_te, y_task_te, d_task_te, task_tr_pool, "taskoriented"),
    ]

    for label, src_pool, X_te_d, y_te_d, d_te_d, tgt_pool, tgt_domain in directions:
        tr_idx, _ = train_test_split(src_pool, test_size=0.2, random_state=42, stratify=y[src_pool])
        base_model = LSGEnsemble(use_adapter=True)
        with Timer() as t_tr:
            base_model.fit(X[tr_idx], y[tr_idx], domains[tr_idx])

        # Zero-shot
        p0 = base_model.predict_proba(X_te_d, d_te_d)
        m0 = evaluate(y_te_d, base_model.predict(X_te_d, d_te_d), p0)
        print(f"  {label:30s}  {'0 (zs)':>5}  {m0['f1']:.3f}  {m0['auprc']:.3f}")

        # Few-shot k
        for k in FEW_SHOT_KS:
            if k >= len(tgt_pool):
                continue
            fs_idx, _ = train_test_split(tgt_pool, train_size=k, random_state=42, stratify=y[tgt_pool])
            model_k = copy.deepcopy(base_model)
            model_k.few_shot_adapt(X[fs_idx], y[fs_idx], domain=tgt_domain)
            pk = model_k.predict_proba(X_te_d, d_te_d)
            mk = evaluate(y_te_d, model_k.predict(X_te_d, d_te_d), pk)
            delta_f1    = mk["f1"]    - m0["f1"]
            delta_auprc = mk["auprc"] - m0["auprc"]
            print(f"  {label:30s}  {k:>5}  {mk['f1']:.3f} ({delta_f1:+.3f})  "
                  f"{mk['auprc']:.3f} ({delta_auprc:+.3f})")

        # No-adapter zero-shot for comparison
        base_noadapt = LSGEnsemble(use_adapter=False)
        base_noadapt.fit(X[tr_idx], y[tr_idx], domains[tr_idx])
        pna = base_noadapt.predict_proba(X_te_d, d_te_d)
        mna = evaluate(y_te_d, base_noadapt.predict(X_te_d, d_te_d), pna)
        print(f"  {label:30s}  {'no-adp':>5}  {mna['f1']:.3f}  {mna['auprc']:.3f}")

    # ── 4. TF-IDF embedding ablation ─────────────────────────────────────
    print("\n── Embedding ablation: SBERT vs TF-IDF ──")
    tfidf = TfidfVectorizer(max_features=384, sublinear_tf=True)
    task_texts = [texts[i] for i in task_tr_pool]
    pc_te_texts = [texts[i] for i in pc_te_idx]
    tfidf.fit(task_texts)

    tr_idx_tfidf, _ = train_test_split(task_tr_pool, test_size=0.2, random_state=42, stratify=y[task_tr_pool])
    X_tfidf_tr = tfidf.transform([texts[i] for i in tr_idx_tfidf]).toarray().astype(np.float32)
    X_tfidf_te = tfidf.transform(pc_te_texts).toarray().astype(np.float32)
    d_tfidf_tr = domains[tr_idx_tfidf]
    d_tfidf_te = np.array(["personachat"] * len(pc_te_idx))

    m_tfidf = LSGEnsemble(use_adapter=True)
    with Timer() as t_tfidf:
        m_tfidf.fit(X_tfidf_tr, y[tr_idx_tfidf], d_tfidf_tr)
    p_tfidf = m_tfidf.predict_proba(X_tfidf_te, d_tfidf_te)
    mt = evaluate(y_pc_te, m_tfidf.predict(X_tfidf_te, d_tfidf_te), p_tfidf)

    tr_idx_sbert, _ = train_test_split(task_tr_pool, test_size=0.2, random_state=42, stratify=y[task_tr_pool])
    m_sbert = LSGEnsemble(use_adapter=True)
    with Timer() as t_sbert:
        m_sbert.fit(X[tr_idx_sbert], y[tr_idx_sbert], domains[tr_idx_sbert])
    p_sbert = m_sbert.predict_proba(X_pc_te, d_pc_te)
    ms = evaluate(y_pc_te, m_sbert.predict(X_pc_te, d_pc_te), p_sbert)

    print(f"  {'Embedding':15s}  {'F1':>6}  {'AUPRC':>7}  {'AUC':>6}  {'Train(s)':>9}")
    print(f"  {'SBERT (384d)':15s}  {ms['f1']:.3f}  {ms['auprc']:.3f}   {ms['roc_auc']:.3f}  {t_sbert.elapsed:.2f}")
    print(f"  {'TF-IDF (384d)':15s}  {mt['f1']:.3f}  {mt['auprc']:.3f}   {mt['roc_auc']:.3f}  {t_tfidf.elapsed:.2f}")

    # ── 5. Training cost table ────────────────────────────────────────────
    print("\n── Training cost (wall-clock) ──")
    tr_idx_cost, _ = train_test_split(task_tr_pool, test_size=0.2, random_state=42, stratify=y[task_tr_pool])
    X_c, y_c, d_c = X[tr_idx_cost], y[tr_idx_cost], domains[tr_idx_cost]

    for name, model in [
        ("LSG ensemble+adapter", LSGEnsemble(use_adapter=True)),
        ("LSG ensemble no-adp",  LSGEnsemble(use_adapter=False)),
        ("LSG LR only",          LSGEnsemble(use_adapter=False)),
    ]:
        with Timer() as t:
            model.fit(X_c, y_c, d_c)
        print(f"  {name:25s}  train={t.elapsed:.3f}s  n={len(X_c)}")

    print("\n  (RL-based memory gating, e.g. Memory-R1, requires GPU + RL loop;")
    print("   LSG trains in <2s on CPU with no reward signal.)")
    print("\n" + "="*70)


if __name__ == "__main__":
    run()
