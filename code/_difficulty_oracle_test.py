"""Is the trace classifier a correctness predictor or just a difficulty gauge?

Compare, for predicting per-trace correctness (AP):
  - trace classifier: canonical_no_sr predicted_prob (LOPO CV, un_augmented)
  - DIFFICULTY ORACLE: leave-one-model-out cohort accuracy per puzzle (peer success;
    not available at solve time) -> a very strong baseline that wins the extremes for free.

Full set is dominated by all-agree puzzles (oracle wins those trivially). The CONTESTED
middle (some models right, some wrong) is where difficulty alone is ~uninformative and only
a trace-reading classifier can say WHICH trace succeeded.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "data" / "insight_4domain"
INSIGHT = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]

preds = pd.read_parquet(PAPER / "predictions.parquet")
preds["pass_type"] = preds["row_id"].str.rsplit("|", n=1).str[-1]
clf = preds[(preds["config"] == "a") & (preds["feature_subset"] == "canonical_no_sr") &
            (preds["cv_setup"] == "lopo_within_domain") & (preds["pass_type"] == "un_augmented")]
clf = clf[["row_id", "domain", "model", "puzzle_id", "predicted_prob", "true_label"]].copy()

# leave-one-model-out cohort accuracy per (domain, puzzle)
g = clf.groupby(["domain", "puzzle_id"])["true_label"]
clf["n_puz"] = g.transform("count")
clf["k_puz"] = g.transform("sum")
clf["oracle"] = (clf["k_puz"] - clf["true_label"]) / (clf["n_puz"] - 1).clip(lower=1)

clf = clf[clf["domain"].isin(INSIGHT)]


def ap_pair(d, label):
    y = d["true_label"].values
    if y.sum() == 0 or y.sum() == len(y):
        print(f"{label:34s} | (degenerate: base rate {y.mean():.2f})"); return
    ap_clf = average_precision_score(y, d["predicted_prob"].values)
    ap_ora = average_precision_score(y, d["oracle"].values)
    print(f"{label:34s} | n={len(d):5d} base={y.mean():.2f} | "
          f"classifier AP {ap_clf:.3f} | difficulty-oracle AP {ap_ora:.3f} | "
          f"clf-oracle {ap_clf-ap_ora:+.3f}")


def main():
    print("Predicting per-trace correctness. AP; higher = better. Insight, un_augmented.\n")
    print("=== FULL SET ===")
    ap_pair(clf, "pooled")
    for dom in INSIGHT:
        ap_pair(clf[clf.domain == dom], f"  {dom}")

    print("\n=== CONTESTED MIDDLE (0 < models-correct < all on the puzzle) ===")
    mid = clf[(clf["k_puz"] > 0) & (clf["k_puz"] < clf["n_puz"])]
    ap_pair(mid, "pooled (contested)")
    for dom in INSIGHT:
        ap_pair(mid[mid.domain == dom], f"  {dom}")

    print("\n=== NARROW MIDDLE (cohort accuracy 0.3-0.7) ===")
    acc = clf["k_puz"] / clf["n_puz"]
    narrow = clf[(acc >= 0.3) & (acc <= 0.7)]
    ap_pair(narrow, "pooled (0.3-0.7)")
    print("\nOn contested puzzles the oracle gives every trace ~the same peer number, so if the")
    print("classifier's AP exceeds the oracle's, it is reading per-trace signal, not just difficulty.")


if __name__ == "__main__":
    main()
