"""Classifier performance with the feature-selection sample removed.

Reviewer KLd3 asks for the classifier to be tested "on data that the dictionary
didn't have contact with". Three of the four domains satisfy that by collection
order. Rebus does not: the marker lists were drawn up from 20 of its 100
puzzles.

This drops those 20 entirely and reports what the classifier does on the
remaining 80, under both evaluations:

  LODO   train on the other three domains, test on Rebus. The 20 are never in
         training under this split, so removing them from the test set isolates
         evaluation contamination.
  LOPO   train and test within Rebus. Here the 20 are in the training folds too,
         so we drop them from both.

Base rates differ between the subsets (the 20 are easier), so lift is reported
alongside AP.

    .venv-x64/Scripts/python.exe llm_evaluation/datasets/_holdout_selection_sample.py

Ported from the working repository for the public release. Paths resolve through
_paths.py against data/ in this bundle; the analysis itself is unchanged.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (PAPER, JUDGE_DIR, AUDIT, EFFORT, DOMAINS, GEMINI_FAMILY,
                    JUDGE_MODEL, SECOND_JUDGE_EXTENDED, selection_puzzles,
                    load_traces_with_features, chosen_conf_table)

HERE = Path(__file__).resolve().parent

EFFORT = ["tokens_thinking_proxy", "elapsed", "hedge_position_variance",
          "thinking_char_count", "hedge_rate", "hedge_ratio"]
DOMAINS = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]

def fit_predict(train, test):
    sc = StandardScaler().fit(train[EFFORT].fillna(0))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(
        sc.transform(train[EFFORT].fillna(0)), train.correct)
    return lr.predict_proba(sc.transform(test[EFFORT].fillna(0)))[:, 1]

def lopo(frame):
    probs = np.zeros(len(frame))
    y = frame.correct.values
    for pid in frame.puzzle_id.unique():
        m = (frame.puzzle_id == pid).values
        if y[~m].std() == 0:
            probs[m] = y[~m].mean()
            continue
        probs[m] = fit_predict(frame[~m], frame[m])
    return probs

def line(label, y, p):
    base = float(np.mean(y))
    ap = average_precision_score(y, p)
    lift = (ap - base) / max(1 - base, 1e-9)
    print(f"  {label:<44}{len(y):>6}{base:>8.3f}{ap:>8.3f}{lift:>8.3f}")
    return ap

def main():
    seen = selection_puzzles()
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    traces["puzzle_id"] = traces.puzzle_id.astype(str)
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = traces[["row_id", "domain", "model", "puzzle_id", "answer", "correct"]].merge(
        feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["answer"].fillna("").str.len() > 0].reset_index(drop=True)

    rebus = df[df.domain == "Rebus"].copy()
    rebus["seen"] = rebus.puzzle_id.isin(seen)
    clean = rebus[~rebus.seen].reset_index(drop=True)
    print(f"Rebus: {rebus.puzzle_id.nunique()} puzzles, {len(rebus)} rows; "
          f"selection sample {len(seen)} puzzles, {int(rebus.seen.sum())} rows\n")

    print(f"  {'':<44}{'n':>6}{'base':>8}{'AP':>8}{'lift':>8}")
    print("  " + "-" * 72)

    # LODO: one model trained off-Rebus, evaluated on each Rebus subset.
    off = df[df.domain != "Rebus"]
    rebus["p_lodo"] = fit_predict(off, rebus)
    print("  LODO, trained on the other three domains")
    ap_all = line("all 100 Rebus puzzles (published)", rebus.correct.values, rebus.p_lodo.values)
    line("the 20 used for marker selection", rebus[rebus.seen].correct.values,
         rebus[rebus.seen].p_lodo.values)
    ap_80 = line("the 80 never used (uncontaminated)", clean.correct.values,
                 rebus[~rebus.seen].p_lodo.values)
    print(f"  {'delta, 80 minus all 100':<44}{'':>6}{'':>8}{ap_80 - ap_all:>+8.3f}\n")

    # LOPO: retrain within Rebus with the 20 removed from training as well.
    print("  LOPO, trained and tested within Rebus")
    rebus["p_lopo"] = lopo(rebus)
    ap_lopo_all = line("all 100 puzzles (published)", rebus.correct.values, rebus.p_lopo.values)
    ap_lopo_80 = line("the 80, with the 20 dropped from training too",
                      clean.correct.values, lopo(clean))
    print(f"  {'delta, 80 minus all 100':<44}{'':>6}{'':>8}{ap_lopo_80 - ap_lopo_all:>+8.3f}\n")

    # Four-domain LODO mean, with Rebus restricted to the clean 80.
    print("  Four-domain LODO mean")
    aps_pub, aps_clean = [], []
    for dom in DOMAINS:
        tr, te = df[df.domain != dom], df[df.domain == dom]
        p = fit_predict(tr, te)
        aps_pub.append(average_precision_score(te.correct, p))
        if dom == "Rebus":
            aps_clean.append(ap_80)
        else:
            aps_clean.append(aps_pub[-1])
    print(f"    published (Rebus = all 100)      mean AP {np.mean(aps_pub):.3f}   "
          f"{', '.join(f'{d} {a:.3f}' for d, a in zip(DOMAINS, aps_pub))}")
    print(f"    Rebus restricted to the clean 80 mean AP {np.mean(aps_clean):.3f}   "
          f"{', '.join(f'{d} {a:.3f}' for d, a in zip(DOMAINS, aps_clean))}")

if __name__ == "__main__":
    main()
