"""Does the hedge dictionary help MORE on the traces it was authored on?

The plain seen-20 vs unseen-80 comparison (_hedge_dictionary_exposure.py) is
confounded: the 20 puzzles the dictionary was written against are systematically
easier (base rate 0.713 vs 0.605), which inflates their AP whatever the features.

This isolates the dictionary's own contribution by differencing:

    increment = AP(6 features) - AP(3 dictionary-free effort features)

computed separately on the seen-20 and the unseen-80. The three dictionary-free
features (tokens_thinking_proxy, elapsed, thinking_char_count) absorb puzzle
difficulty, so the increment measures what the hedge dictionary adds on top.

If the dictionary were fitted to the traces it was authored on, its increment
should be larger on the seen-20 than on the unseen-80. A puzzle-clustered
bootstrap puts a CI on that difference.

    .venv-x64/Scripts/python.exe llm_evaluation/datasets/_hedge_dictionary_increment.py

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

DICT_FREE = ["tokens_thinking_proxy", "elapsed", "thinking_char_count"]
FULL_SIX = DICT_FREE + ["hedge_rate", "hedge_ratio", "hedge_position_variance"]
N_BOOT = 2000
SEED = 42

def fit_predict(train, test, feats):
    sc, lr = StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.fit_transform(train[feats].fillna(0).values), train.correct.values)
    return lr.predict_proba(sc.transform(test[feats].fillna(0).values))[:, 1]

def main():
    seen = selection_puzzles()

    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    traces["puzzle_id"] = traces.puzzle_id.astype(str)
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = traces[["row_id", "domain", "model", "puzzle_id", "answer", "correct"]
                ].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["answer"].fillna("").str.len() > 0].reset_index(drop=True)

    rebus = df[df.domain == "Rebus"].copy().reset_index(drop=True)
    rebus["seen"] = rebus.puzzle_id.isin(seen)
    off_rebus = df[df.domain != "Rebus"]

    # LODO: one model trained entirely off-Rebus, so neither Rebus subset is in training.
    for feats, name in ((DICT_FREE, "p3"), (FULL_SIX, "p6")):
        rebus[name] = fit_predict(off_rebus, rebus, feats)

    rng = np.random.default_rng(SEED)
    pids = {s: rebus.loc[rebus.seen == s, "puzzle_id"].unique() for s in (True, False)}
    idx_by_pid = {p: np.flatnonzero((rebus.puzzle_id == p).values) for p in rebus.puzzle_id.unique()}

    def increment(rows):
        y = rebus.correct.values[rows]
        if y.std() == 0:
            return np.nan
        return (average_precision_score(y, rebus.p6.values[rows])
                - average_precision_score(y, rebus.p3.values[rows]))

    print(f"{'subset':<12}{'n':>6}{'base':>8}{'AP(3 dict-free)':>18}{'AP(6)':>9}{'increment':>12}")
    print("-" * 65)
    point = {}
    for s, label in ((True, "seen-20"), (False, "unseen-80")):
        rows = np.flatnonzero((rebus.seen == s).values)
        y = rebus.correct.values[rows]
        ap3 = average_precision_score(y, rebus.p3.values[rows])
        ap6 = average_precision_score(y, rebus.p6.values[rows])
        point[s] = ap6 - ap3
        print(f"{label:<12}{len(rows):>6}{y.mean():>8.3f}{ap3:>18.3f}{ap6:>9.3f}{ap6 - ap3:>+12.3f}")

    diff = point[True] - point[False]
    print(f"\ndifference in increment (seen minus unseen): {diff:+.4f}")

    boots = []
    for _ in range(N_BOOT):
        vals = {}
        for s in (True, False):
            drawn = rng.choice(pids[s], size=len(pids[s]), replace=True)
            rows = np.concatenate([idx_by_pid[p] for p in drawn])
            vals[s] = increment(rows)
        if not (np.isnan(vals[True]) or np.isnan(vals[False])):
            boots.append(vals[True] - vals[False])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"95% CI (puzzle-clustered bootstrap, N={len(boots)}): [{lo:+.4f}, {hi:+.4f}]")
    print("\nIf the CI spans zero, there is no evidence the dictionary helps more on")
    print("the traces it was authored on than on the 80 puzzles it never saw.")

if __name__ == "__main__":
    main()
