"""Effort-signal classifier (6 features) vs LEN+RATES (21 features).

Step 3b: does shrinking to the durable feature set cost much accuracy?
Success criterion: <5pp AUC cost on transfer (LODO).

Two evaluations:
  1. Within-domain LOPO per domain (4 numbers per feature set)
  2. LODO: train on 3 domains, test on the held-out 4th (4 numbers per set)

Feature sets compared:
  - EFFORT_SIGNAL (6): the 3 strict-universal + 3 nearly-universal features
  - LEN_RATES (21): the canonical no-SR feature set
  - EFFORT_SIGNAL_WITH_SR (8): effort signal + self_reported_conf + present
  - LEN_RATES_WITH_SR (23): canonical with-SR

Reports AUC, AP, and base-rate-adjusted lift; deltas vs LEN+RATES.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PAPER = Path(__file__).resolve().parent.parent / "data" / "insight_4domain"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _classifier_sweep import (CANONICAL_LEN_RATES_NO_SR,  # noqa: E402
                                CANONICAL_LEN_RATES_WITH_SR)

EFFORT_SIGNAL = [
    "tokens_thinking_proxy",
    "elapsed",
    "hedge_position_variance",
    "thinking_char_count",
    "hedge_rate",
    "hedge_ratio",
]
EFFORT_SIGNAL_WITH_SR = ["self_reported_conf", "self_reported_present"] + EFFORT_SIGNAL


def fit_and_score(Xtr, ytr, Xte, yte):
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.fit_transform(Xtr), ytr)
    p = lr.predict_proba(sc.transform(Xte))[:, 1]
    base = float(yte.mean())
    if yte.std() == 0:
        return None
    auc = roc_auc_score(yte, p)
    ap = average_precision_score(yte, p)
    lift = (ap - base) / max(1 - base, 1e-9)
    return {"AUC": auc, "AP": ap, "lift": lift, "base": base}


def lopo_within_domain(df, features, domain):
    sub = df[df["domain"] == domain].copy()
    pids = sorted(sub["puzzle_id"].unique())
    feats = sub[features].fillna(0).values
    y = sub["correct"].values
    probs = np.zeros(len(sub))
    for pid in pids:
        mask = (sub["puzzle_id"] == pid).values
        Xtr, Xte = feats[~mask], feats[mask]
        ytr = y[~mask]
        if ytr.std() == 0:
            probs[mask] = ytr.mean()
            continue
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(sc.fit_transform(Xtr), ytr)
        probs[mask] = lr.predict_proba(sc.transform(Xte))[:, 1]
    if y.std() == 0:
        return None
    base = float(y.mean())
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    lift = (ap - base) / max(1 - base, 1e-9)
    return {"AUC": auc, "AP": ap, "lift": lift, "base": base, "n": len(sub)}


def lodo(df, features, holdout):
    train = df[df["domain"] != holdout]
    test = df[df["domain"] == holdout]
    if len(train) == 0 or len(test) == 0:
        return None
    Xtr = train[features].fillna(0).values
    Xte = test[features].fillna(0).values
    ytr = train["correct"].values
    yte = test["correct"].values
    res = fit_and_score(Xtr, ytr, Xte, yte)
    if res is None:
        return None
    res.update({"n_train": len(train), "n_test": len(test)})
    return res


def print_block(title, results, baseline_results, domains):
    """Print a block: results per domain with delta-AUC vs baseline."""
    print(title)
    print(f"  {'Domain':<16s} {'n':>6s} {'base':>5s} {'AP':>6s} {'AUC':>6s} {'ΔAUC':>7s}")
    deltas = []
    for d in domains:
        r = results.get(d)
        b = baseline_results.get(d)
        if r is None:
            continue
        delta_auc = r["AUC"] - b["AUC"] if b else float("nan")
        deltas.append(delta_auc)
        n = r.get("n_test", r.get("n", 0))
        print(f"  {d:<16s} {n:>6d} {r['base']:>5.3f} {r['AP']:>6.3f} {r['AUC']:>6.3f}  {delta_auc:>+6.3f}")
    if deltas:
        mean_delta = np.mean([d for d in deltas if not np.isnan(d)])
        print(f"  {'mean':<16s} {'':>6s} {'':>5s} {'':>6s} {'':>6s}  {mean_delta:>+6.3f}")


def main():
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = traces[["row_id", "domain", "model", "puzzle_id", "answer", "correct"]
                ].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["answer"].fillna("").str.len() > 0].reset_index(drop=True)
    domains = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]

    feature_sets = [
        ("LEN+RATES (21, baseline no-SR)", CANONICAL_LEN_RATES_NO_SR),
        ("EFFORT_SIGNAL (6, no-SR)",      EFFORT_SIGNAL),
        ("LEN+RATES+SR (23, baseline with-SR)", CANONICAL_LEN_RATES_WITH_SR),
        ("EFFORT_SIGNAL+SR (8, with-SR)", EFFORT_SIGNAL_WITH_SR),
    ]

    # ─────────────────────────────────────────────────────────────
    # 1. Within-domain LOPO
    # ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print("WITHIN-DOMAIN LOPO (per-domain classifiers)")
    print("=" * 80)
    lopo_results = {}
    for label, feats in feature_sets:
        per_d = {}
        for d in domains:
            per_d[d] = lopo_within_domain(df, feats, d)
        lopo_results[label] = per_d

    # Print baseline first, then minimal sets with delta vs their baseline
    baseline_no_sr = lopo_results["LEN+RATES (21, baseline no-SR)"]
    baseline_w_sr = lopo_results["LEN+RATES+SR (23, baseline with-SR)"]
    for label, _ in feature_sets:
        is_baseline = label.startswith("LEN+RATES")
        if "+SR" in label and "EFFORT" in label:
            base_for_delta = baseline_w_sr
        elif "EFFORT" in label:
            base_for_delta = baseline_no_sr
        elif "+SR" in label:
            base_for_delta = baseline_w_sr  # zero delta with itself
        else:
            base_for_delta = baseline_no_sr
        print()
        print_block(label, lopo_results[label], base_for_delta, domains)

    # ─────────────────────────────────────────────────────────────
    # 2. LODO (cross-domain)
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("LODO (train 3 domains, test on held-out 4th)")
    print("=" * 80)
    lodo_results = {}
    for label, feats in feature_sets:
        per_d = {}
        for d in domains:
            per_d[d] = lodo(df, feats, d)
        lodo_results[label] = per_d

    baseline_no_sr = lodo_results["LEN+RATES (21, baseline no-SR)"]
    baseline_w_sr = lodo_results["LEN+RATES+SR (23, baseline with-SR)"]
    for label, _ in feature_sets:
        if "+SR" in label and "EFFORT" in label:
            base_for_delta = baseline_w_sr
        elif "EFFORT" in label:
            base_for_delta = baseline_no_sr
        elif "+SR" in label:
            base_for_delta = baseline_w_sr
        else:
            base_for_delta = baseline_no_sr
        print()
        print_block(label, lodo_results[label], base_for_delta, domains)

    # ─────────────────────────────────────────────────────────────
    # 3. Verdict
    # ─────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("VERDICT (success criterion: |ΔAUC| < 5pp on transfer)")
    print("=" * 80)
    for label_min, label_base in [
        ("EFFORT_SIGNAL (6, no-SR)",      "LEN+RATES (21, baseline no-SR)"),
        ("EFFORT_SIGNAL+SR (8, with-SR)", "LEN+RATES+SR (23, baseline with-SR)"),
    ]:
        deltas_lopo = []
        deltas_lodo = []
        for d in domains:
            rm = lopo_results[label_min].get(d)
            rb = lopo_results[label_base].get(d)
            if rm and rb:
                deltas_lopo.append(rm["AUC"] - rb["AUC"])
            rm = lodo_results[label_min].get(d)
            rb = lodo_results[label_base].get(d)
            if rm and rb:
                deltas_lodo.append(rm["AUC"] - rb["AUC"])
        print(f"\n{label_min} vs {label_base}:")
        print(f"  LOPO mean ΔAUC: {np.mean(deltas_lopo):+.3f}  (range {min(deltas_lopo):+.3f} to {max(deltas_lopo):+.3f})")
        print(f"  LODO mean ΔAUC: {np.mean(deltas_lodo):+.3f}  (range {min(deltas_lodo):+.3f} to {max(deltas_lodo):+.3f})")
        worst = min(deltas_lodo)
        verdict = "PASS" if worst > -0.05 else "FAIL"
        print(f"  worst-case LODO ΔAUC: {worst:+.3f}  -> {verdict} (threshold -0.050)")


if __name__ == "__main__":
    main()
