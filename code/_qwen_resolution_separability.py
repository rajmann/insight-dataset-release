"""A3: does Qwen3-VL specifically lack the RESOLUTION-event signature, while keeping
the shared LEVEL/effort difficulty signal?

Per model, measure correct-vs-wrong separation of single features, split into:
  RESOLUTION (dynamics: did a discrete commit event happen, and where in the trace)
    hedge_shift, hedge_density_last_third, hedge_position_variance, confidence_rate
  LEVEL (effort magnitude: the shared difficulty signal - wrong traces longer/hedgier)
    hedge_rate, thinking_char_count, elapsed, tokens_thinking_proxy

Separation = |AUC - 0.5| of the single feature predicting correctness, computed WITHIN
each (model, domain) then averaged across the 4 insight domains (scale-free, robust to
Qwen's saturated feature distributions). AUC (not Cohen's d) because Qwen's counts saturate.

Prediction (scope-condition hypothesis): Qwen keeps LEVEL separation but collapses on
RESOLUTION separation. If Qwen is flat on BOTH, it is simply a low-signal model and the
resolution-specific story is wrong.

Insight domains, un_augmented. Data: all_traces.parquet (correct) + features.parquet.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "data" / "insight_4domain"
INSIGHT = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]

RESOLUTION = ["hedge_shift", "hedge_density_last_third", "hedge_position_variance", "confidence_rate"]
LEVEL = ["hedge_rate", "thinking_char_count", "elapsed", "tokens_thinking_proxy"]
ALL_FEATS = RESOLUTION + LEVEL


def load():
    tr = pd.read_parquet(PAPER / "all_traces.parquet")
    tr = tr[(tr["pass_type"] == "un_augmented") & (tr["answer"].fillna("").str.len() > 0)]
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = tr[["row_id", "domain", "model", "puzzle_id", "correct"]].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    return df[df["domain"].isin(INSIGHT)].reset_index(drop=True)


def sep_within(df, model, feat):
    """Mean over domains of |AUC-0.5| for `feat` predicting correctness, within (model,domain)."""
    seps = []
    for dom in INSIGHT:
        sub = df[(df.model == model) & (df.domain == dom)].dropna(subset=[feat])
        y = sub["correct"].values
        if len(sub) < 15 or y.sum() < 3 or y.sum() > len(y) - 3:
            continue
        x = sub[feat].values.astype(float)
        if np.nanstd(x) == 0:
            seps.append(0.0); continue
        auc = roc_auc_score(y, x)
        seps.append(abs(auc - 0.5))
    return float(np.mean(seps)) if seps else float("nan")


def main():
    df = load()
    models = sorted(df["model"].unique())
    print("A3: correct-vs-wrong feature separation |AUC-0.5| (0=none, 0.5=perfect).")
    print("Averaged within (model,domain) across 4 insight domains. un_augmented.\n")

    # per-model separation on each group
    rows = {}
    for m in models:
        res = np.nanmean([sep_within(df, m, f) for f in RESOLUTION])
        lev = np.nanmean([sep_within(df, m, f) for f in LEVEL])
        rows[m] = (res, lev)

    def short(m):
        return (m.replace("Qwen3-VL-235B Thinking", "Qwen3-VL").replace("Claude ", "")
                 .replace("Gemini ", "Gem ").replace(" Thinking", ""))

    hdr = f"{'model':16s} | {'RESOLUTION':>10s} | {'LEVEL':>7s} | {'res - lev':>9s}"
    print(hdr); print("-" * len(hdr))
    # print Qwen last for emphasis; others sorted by resolution sep desc
    others = sorted([m for m in models if "Qwen" in m.upper() and False] + [m for m in models if "Qwen" not in m], key=lambda m: -rows[m][0])
    qwen = [m for m in models if "Qwen" in m]
    for m in others + qwen:
        res, lev = rows[m]
        print(f"{short(m):16s} | {res:10.3f} | {lev:7.3f} | {res - lev:+9.3f}")
    print("-" * len(hdr))
    others_only = [m for m in models if "Qwen" not in m]
    res_o = np.nanmean([rows[m][0] for m in others_only])
    lev_o = np.nanmean([rows[m][1] for m in others_only])
    print(f"{'mean(non-Qwen)':16s} | {res_o:10.3f} | {lev_o:7.3f} | {res_o - lev_o:+9.3f}")

    # per-feature detail for Qwen vs others
    print("\nPer-feature separation (Qwen vs mean of non-Qwen):")
    print(f"{'feature':26s} | {'group':>10s} | {'Qwen':>6s} | {'others':>6s} | {'Q-oth':>6s}")
    print("-" * 66)
    for f in ALL_FEATS:
        grp = "RESOLUTION" if f in RESOLUTION else "LEVEL"
        q = np.nanmean([sep_within(df, m, f) for m in qwen])
        o = np.nanmean([sep_within(df, m, f) for m in others_only])
        print(f"{f:26s} | {grp:>10s} | {q:6.3f} | {o:6.3f} | {q - o:+6.3f}")
    print("\nHypothesis holds if Qwen's RESOLUTION separation is near 0 while LEVEL is retained,")
    print("and non-Qwen models separate on RESOLUTION. Flat-on-both would refute it.")

    # ---- Reconciliation: within-model 6-feature LOPO classifier, AUC (base-rate-invariant)
    #      vs AP (base-rate-sensitive). Shows whether low classifier AP is a base-rate artefact.
    EFFORT6 = ["tokens_thinking_proxy", "elapsed", "hedge_position_variance",
               "thinking_char_count", "hedge_rate", "hedge_ratio"]
    def lopo_within(sub):
        """Within (already domain-filtered) LOPO probs for the 6-feature effort classifier."""
        y = sub["correct"].values
        X = sub[EFFORT6].values.astype(float)
        pids = sub["puzzle_id"].values
        probs = np.zeros(len(sub))
        for p in np.unique(pids):
            te = pids == p
            if (~te).sum() < 10 or y[~te].std() == 0:
                probs[te] = y[~te].mean() if (~te).sum() else y.mean(); continue
            sc = StandardScaler(); lr = LogisticRegression(max_iter=2000, C=1.0)
            lr.fit(sc.fit_transform(X[~te]), y[~te])
            probs[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        return y, probs

    print("\n\nWithin-(model,domain) 6-feature effort classifier, LOPO; AUC/AP averaged over domains.")
    print("AUC is base-rate-invariant; AP is base-rate-bounded. (No cross-domain leakage.)")
    print(f"{'model':16s} | {'base':>5s} | {'AUC':>6s} | {'AP':>6s} | {'AP-base':>7s}")
    print("-" * 52)
    for m in others_only + qwen:
        aucs, aps, bases = [], [], []
        for dom in INSIGHT:
            sub = df[(df.model == m) & (df.domain == dom)].dropna(subset=EFFORT6).reset_index(drop=True)
            y = sub["correct"].values
            if len(sub) < 20 or y.sum() < 4 or y.sum() > len(y) - 4:
                continue
            y, probs = lopo_within(sub)
            aucs.append(roc_auc_score(y, probs)); aps.append(average_precision_score(y, probs))
            bases.append(y.mean())
        if not aucs:
            print(f"{short(m):16s} | (insufficient)"); continue
        print(f"{short(m):16s} | {np.mean(bases):5.2f} | {np.mean(aucs):6.3f} | "
              f"{np.mean(aps):6.3f} | {np.mean(aps) - np.mean(bases):+7.3f}")
    print("-" * 52)
    print("If Qwen's AUC is comparable to others but AP tracks its low base rate, the classifier")
    print("'floor' (reported as AUC-PR/AP in the paper) is a base-rate effect, not missing signal.")


if __name__ == "__main__":
    main()
