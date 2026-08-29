"""Benjamini-Hochberg FDR check for Table 12 (per-feature Spearman, 6 features x 5 domains).

Computes a puzzle-clustered bootstrap 2-sided p-value per cell, then applies BH at q=0.05
across the 30 tests. Reports how many originally-significant cells survive.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _math_appendix_spearman import load_math

PD = HERE / "paper_dataset_4domain_2026-05-05"
FEATS = ["tokens_thinking_proxy", "elapsed", "thinking_char_count",
         "hedge_rate", "hedge_ratio", "hedge_position_variance"]
RNG = np.random.default_rng(0)
NB = 2000


def boot_p(df, feat):
    d = df.dropna(subset=[feat])
    clusters = {k: idx.values for k, idx in d.groupby("puzzle_id").groups.items()}
    keys = np.array(list(clusters))
    x = d[feat].astype(float); y = d["correct"].astype(float)
    rho = spearmanr(x, y).statistic
    pos = neg = 0
    for _ in range(NB):
        pick = keys[RNG.integers(0, len(keys), len(keys))]
        idx = np.concatenate([clusters[k] for k in pick])
        xx = x.loc[idx].values; yy = y.loc[idx].values
        if np.std(xx) == 0 or np.std(yy) == 0:
            continue
        r = spearmanr(xx, yy).statistic
        if r > 0: pos += 1
        elif r < 0: neg += 1
    p = 2 * min(pos, neg) / NB
    return rho, max(p, 1.0 / NB)   # floor at 1/NB


def main():
    # insight: features + outcomes, un-augmented, 7-model cohort
    feats = pd.read_parquet(PD / "features.parquet")
    tr = pd.read_parquet(PD / "all_traces.parquet")[["row_id", "domain", "puzzle_id", "correct", "pass_type"]]
    ins = feats.merge(tr, on="row_id", how="inner")
    ins = ins[ins.pass_type == "un_augmented"]
    math = load_math()

    cells = []   # (feature, domain, rho, p)
    for dom in ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]:
        d = ins[ins.domain == dom]
        for f in FEATS:
            rho, p = boot_p(d, f)
            cells.append((f, dom, rho, p))
    for f in FEATS:
        rho, p = boot_p(math, f)
        cells.append((f, "Math", rho, p))

    cf = pd.DataFrame(cells, columns=["feature", "domain", "rho", "p"])
    cf["sig_raw"] = cf.p < 0.05
    # Benjamini-Hochberg at q=0.05
    m = len(cf); q = 0.05
    cf = cf.sort_values("p").reset_index(drop=True)
    cf["rank"] = cf.index + 1
    cf["bh_thresh"] = cf["rank"] / m * q
    kmax = cf.index[cf.p <= cf.bh_thresh].max() if (cf.p <= cf.bh_thresh).any() else -1
    cf["bh_survive"] = cf.index <= kmax

    print(f"m = {m} tests, q = {q}\n")
    print(cf.sort_values(["domain", "feature"])[["feature", "domain", "rho", "p", "sig_raw", "bh_survive"]]
          .to_string(index=False))
    print(f"\nRaw significant (p<0.05): {cf.sig_raw.sum()}")
    print(f"Survive BH-FDR (q=0.05):  {cf.bh_survive.sum()}")
    disagree = cf[cf.sig_raw != cf.bh_survive]
    print(f"Changed by BH: {len(disagree)}"
          + ("" if not len(disagree) else "\n" + disagree[['feature','domain','rho','p']].to_string(index=False)))


if __name__ == "__main__":
    main()
