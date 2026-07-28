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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lsg.data import load_taskoriented, load_personachat
from lsg.embeddings import encode
from lsg.baseline import StatisticalGate
from lsg.model import LSGEnsemble
from lsg.evaluate import evaluate, Timer

FEW_SHOT_N = 100        # labeled target-domain examples for adaptation
ADAPT_CURVE_NS = [10, 25, 50, 100, 200]   # sweep for adaptation curve


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
        base_preds = (base_probs >= 0.5).astype(int)
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
            preds = (probs >= 0.5).astype(int)
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
            preds = (probs >= 0.5).astype(int)
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
    m0 = evaluate(y_te, (p0 >= 0.5).astype(int), p0)
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
        m = evaluate(y_te, (probs >= 0.5).astype(int), probs)
        rows.append({"N": n, **{k: round(m[k], 4) for k in ("f1", "roc_auc", "ece")}})
        print(f"  N={n:>3}              F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}  ECE={m['ece']:.3f}")

    # in-domain ceiling
    pc_id_tr, pc_id_te = splits["pc_id_tr"], splits["pc_id_te"]
    ceil_model = LSGEnsemble(use_adapter=True)
    ceil_model.fit(X[pc_id_tr], y[pc_id_tr], domains[pc_id_tr])
    pc = ceil_model.predict_proba(X[pc_id_te], domains[pc_id_te])
    mc = evaluate(y[pc_id_te], (pc >= 0.5).astype(int), pc)
    rows.append({"N": "ceiling", **{k: round(mc[k], 4) for k in ("f1", "roc_auc", "ece")}})
    print(f"  N=ceiling (in-dom) F1={mc['f1']:.3f}  AUC={mc['roc_auc']:.3f}  ECE={mc['ece']:.3f}")

    df_curve = pd.DataFrame(rows).set_index("N")
    print("\n" + df_curve.to_string())
    print("="*60)
    return df_curve


if __name__ == "__main__":
    run()
