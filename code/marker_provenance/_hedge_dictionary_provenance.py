"""Is the hedge dictionary circular? Ablate the label-derived patterns.

Reviewer KLd3 (INLG 2026): the hedge regex list "appears to have been built or
refined by the authors reading through model traces ... If the phrase list was
tuned while looking at which traces were right vs. wrong, the strong AP numbers
may partly reflect the dictionary being fit to the evaluation labels."

That is partly true. `train_confidence_classifier.HEDGE_PATTERNS` records three
patterns added on 2026-04-21 from n-gram discovery over *Rebus* correctness
labels, and two candidates dropped because they hurt LOMO. `CONFIDENCE_PATTERNS`
(which feeds `hedge_ratio` through its denominator) gained eight the same day.

This script asks whether those label-informed edits are what carries the signal,
by recomputing the three dictionary-dependent features under three dictionaries
and re-running the paper's LODO evaluation on each:

  V0  deployed          - base list + the 2026-04-21 additions (the published run)
  V1  label-blind       - base list only; every 2026-04-21 addition removed.
                          This is the dictionary we would have had if we had
                          never looked at a correctness label.
  V2  V1 + rejects      - V1 plus the two hedge terms that were dropped *because*
                          they hurt LOMO, testing the reverse direction of
                          contact (did the label-based rejection do the work?)

The other three deployed features (tokens_thinking_proxy, elapsed,
thinking_char_count) contain no dictionary and are held fixed throughout.

Note on timing, which is the other half of the answer: the dictionary was frozen
on 2026-04-21. Cryptic and VisualPuzzles were collected 2026-04-24 and
Connections+ on 2026-05-04, so for three of the four domains the LODO test set
did not exist when the dictionary was written.

Usage (needs pyarrow -> use the x64 venv on Windows-on-ARM):
    .venv-x64/Scripts/python.exe llm_evaluation/datasets/_hedge_dictionary_provenance.py

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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (PAPER, JUDGE_DIR, AUDIT, EFFORT, DOMAINS, GEMINI_FAMILY,
                    JUDGE_MODEL, SECOND_JUDGE_EXTENDED, selection_puzzles,
                    load_traces_with_features, chosen_conf_table)

HERE = Path(__file__).resolve().parent

# ── Dictionaries ────────────────────────────────────────────────────────────
# Base = everything that predates the 2026-04-21 n-gram-discovery pass. These
# are ordinary epistemic markers of the kind the hedging literature describes.
HEDGE_BASE = [
    r'\bwait\b', r'\bactually\b', r'\bhmm+\b', r'\bno[,.]', r'\bmaybe\b',
    r'\breconsider\b', r"\blet me think\b", r"\bi'm not sure\b", r'\bhold on\b',
    r"\bthat doesn't seem right\b", r'\bperhaps\b', r'\bon second thought\b',
    r"\blet me reconsider\b", r"\bthat can't be\b", r'\bwait,\b',
]
# Added 2026-04-21 from n-gram discovery over Rebus correctness labels.
HEDGE_ADDED = [
    r'\bmight (indicate|represent|suggest)\b',
    r'\balso wonder if\b',
    r'\bother options\b',
]
# Proposed by the same discovery pass, then dropped because they hurt LOMO.
HEDGE_REJECTED = [r"\blet's check\b", r'\bto be sure\b']

CONF_BASE = [
    r'\bclearly\b', r'\bobviously\b', r'\bmust be\b', r'\bthe answer is\b',
    r'\bdefinitely\b', r"\bso it's\b", r'\bconfident\b', r'\bI think the answer\b',
    r"\bthat's it\b", r'\beureka\b', r'\byes!\b', r'\bof course\b',
]
CONF_ADDED = [
    r'\bperfect\b', r'\bgot it\b', r'\bjumps( out)?\b', r'\bimmediately\b',
    r'\bright!', r'\bright off\b', r'\bspot on\b', r'\bmakes sense\b',
]

VARIANTS = {
    "V0 deployed":      (HEDGE_BASE + HEDGE_ADDED, CONF_BASE + CONF_ADDED),
    "V1 label-blind":   (HEDGE_BASE, CONF_BASE),
    "V2 V1+rejected":   (HEDGE_BASE + HEDGE_REJECTED, CONF_BASE),
}

EFFORT_SIGNAL = [
    "tokens_thinking_proxy", "elapsed", "hedge_position_variance",
    "thinking_char_count", "hedge_rate", "hedge_ratio",
]
DOMAINS = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]

def count_patterns(text: str, patterns) -> int:
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)

def hedge_features(thinking: str, hedge_pats, conf_pats) -> dict:
    """Reimplements train_confidence_classifier.extract_features_B for the
    three dictionary-dependent features, verbatim."""
    if not isinstance(thinking, str) or not thinking:
        return {"hedge_rate": 0.0, "hedge_ratio": 0.0, "hedge_position_variance": 0.0}
    hedge = count_patterns(thinking, hedge_pats)
    conf = count_patterns(thinking, conf_pats)
    n_words = max(len(thinking.lower().split()), 1)

    positions = []
    denom = max(len(thinking), 1)
    for pat in hedge_pats:
        for m in re.finditer(pat, thinking, re.IGNORECASE):
            positions.append(m.start() / denom)
    if len(positions) >= 2:
        mean_p = sum(positions) / len(positions)
        var_p = sum((p - mean_p) ** 2 for p in positions) / len(positions)
    else:
        var_p = 0.0

    return {
        "hedge_rate": hedge / n_words,
        "hedge_ratio": hedge / (hedge + conf + 1),
        "hedge_position_variance": var_p,
    }

def lodo(df, features, holdout):
    train, test = df[df["domain"] != holdout], df[df["domain"] == holdout]
    Xtr, Xte = train[features].fillna(0).values, test[features].fillna(0).values
    ytr, yte = train["correct"].values, test["correct"].values
    sc, lr = StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.fit_transform(Xtr), ytr)
    p = lr.predict_proba(sc.transform(Xte))[:, 1]
    return {"AP": average_precision_score(yte, p), "AUC": roc_auc_score(yte, p),
            "base": float(yte.mean()), "n": len(test)}

def main():
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = traces[["row_id", "domain", "model", "puzzle_id", "answer", "correct", "thinking"]
                ].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["answer"].fillna("").str.len() > 0].reset_index(drop=True)
    print(f"rows: {len(df)}   domains: {df['domain'].value_counts().to_dict()}\n")

    results = {}
    for name, (hedge_pats, conf_pats) in VARIANTS.items():
        recomputed = pd.DataFrame(
            [hedge_features(t, hedge_pats, conf_pats) for t in df["thinking"]],
            index=df.index)

        if name.startswith("V0"):
            # Faithfulness check: recomputing the DEPLOYED dictionary must
            # reproduce the values stored in features.parquet.
            print("Faithfulness check (recomputed V0 vs stored features.parquet):")
            for col in ("hedge_rate", "hedge_ratio", "hedge_position_variance"):
                stored, mine = df[col].fillna(0), recomputed[col]
                r = float(np.corrcoef(stored, mine)[0, 1])
                max_abs = float((stored - mine).abs().max())
                print(f"  {col:<26} pearson r = {r:.5f}   max |diff| = {max_abs:.2e}")
            print()

        d = df.copy()
        for col in recomputed.columns:
            d[col] = recomputed[col]
        results[name] = {dom: lodo(d, EFFORT_SIGNAL, dom) for dom in DOMAINS}

    hdr = f"{'Dictionary':<18}" + "".join(f"{dom:>16}" for dom in DOMAINS) + f"{'mean':>9}"
    print("LODO average precision (train on three domains, test on the held-out fourth)")
    print(hdr)
    print("-" * len(hdr))
    for name, per_dom in results.items():
        aps = [per_dom[d]["AP"] for d in DOMAINS]
        print(f"{name:<18}" + "".join(f"{a:>16.3f}" for a in aps) + f"{np.mean(aps):>9.3f}")

    v0 = np.mean([results["V0 deployed"][d]["AP"] for d in DOMAINS])
    print()
    for name in ("V1 label-blind", "V2 V1+rejected"):
        v = np.mean([results[name][d]["AP"] for d in DOMAINS])
        print(f"  {name:<18} mean AP delta vs deployed: {v - v0:+.4f}")

    print("\nPer-domain delta vs deployed (V1 label-blind):")
    for d in DOMAINS:
        delta = results["V1 label-blind"][d]["AP"] - results["V0 deployed"][d]["AP"]
        tag = "  <- dictionary built on this domain's labels" if d == "Rebus" else ""
        print(f"  {d:<18} {delta:+.4f}{tag}")

if __name__ == "__main__":
    main()
