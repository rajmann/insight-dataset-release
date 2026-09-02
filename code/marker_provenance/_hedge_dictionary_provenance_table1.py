"""V0/V1/V2 hedge-dictionary ablation, in the paper's Table-1 configuration.

Companion to _hedge_dictionary_provenance.py, which runs on every extracted row
and reproduces _4domain_effort_signal_classifier.py exactly (Rebus 0.816,
Cryptic 0.906, VP 0.842, ConnP 0.792).

Table 1 reports different numbers (0.82 / 0.92 / 0.83 / 0.84, mean 0.85). The
"+conf AP" column needs the LLM-judge feature, so both of its columns are
computed on the judge-matched subset - the rows for which chosen_conf exists.
This script rebuilds that subset the way _compute_paper_cis.load() does and
re-runs the ablation there, so the deltas are comparable to the published table.

Run from the repo root (the judge CSVs live there):
    .venv-x64/Scripts/python.exe llm_evaluation/datasets/_hedge_dictionary_provenance_table1.py

Ported from the working repository for the public release. Paths resolve through
_paths.py against data/ in this bundle; the analysis itself is unchanged.
"""

from __future__ import annotations

import json
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
sys.path.insert(0, str(HERE))
from _hedge_dictionary_provenance import (  # noqa: E402
    VARIANTS, EFFORT_SIGNAL, DOMAINS, hedge_features)

def judge_table() -> pd.DataFrame:
    """Rebuilt verbatim from _compute_paper_cis.load()."""
    rc = pd.read_csv(JUDGE_DIR / "eureka_judge_full.csv")
    rc["puzzle_id"] = rc.puzzle_id.astype(str)
    rc = rc[rc.is_chosen == 1].dropna(subset=["judge_confidence"]).copy()
    rc["chosen_conf"] = rc.judge_confidence

    vp = pd.read_csv(JUDGE_DIR / "eureka_judge_vp.csv")
    vp["puzzle_id"] = vp.puzzle_id.astype(str)
    vp = vp[(vp.is_chosen == 1) & (vp.pass_type == "un_augmented")].dropna(
        subset=["judge_confidence"]).copy()
    vp["chosen_conf"] = vp.judge_confidence

    cp = pd.read_csv(JUDGE_DIR / "eureka_judge_connp.csv")
    cp["puzzle_id"] = cp.puzzle_id.astype(str)
    cp = cp[cp.error.isna()].copy()

    def mean_agg(gs_json):
        if pd.isna(gs_json):
            return None
        try:
            gs = json.loads(gs_json)
        except Exception:
            return None
        valid = [s for s in gs if isinstance(s, (int, float))]
        return sum(valid) / len(valid) if valid else None

    cp["chosen_conf"] = [mean_agg(g) for g in cp.group_scores]
    cp["domain"] = "ConnectionsPlus"
    cp = cp.dropna(subset=["chosen_conf"])

    cols = ["domain", "model", "puzzle_id", "chosen_conf"]
    return pd.concat([rc[cols], vp[cols], cp[cols]], ignore_index=True)

def lodo_ap(df, features, holdout):
    train, test = df[df["domain"] != holdout], df[df["domain"] == holdout]
    sc, lr = StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(sc.fit_transform(train[features].fillna(0).values), train["correct"].values)
    p = lr.predict_proba(sc.transform(test[features].fillna(0).values))[:, 1]
    return average_precision_score(test["correct"].values, p)

def main():
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    traces["puzzle_id"] = traces.puzzle_id.astype(str)
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = traces[["row_id", "domain", "model", "puzzle_id", "answer", "correct", "thinking"]
                ].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["answer"].fillna("").str.len() > 0]

    judge = judge_table()
    matched = df.merge(judge, on=["domain", "model", "puzzle_id"], how="inner"
                       ).drop_duplicates(subset=["row_id"]).reset_index(drop=True)
    print(f"all extracted rows : {len(df)}")
    print(f"judge-matched rows : {len(matched)}   {matched['domain'].value_counts().to_dict()}\n")

    results = {}
    for name, (hedge_pats, conf_pats) in VARIANTS.items():
        rec = pd.DataFrame([hedge_features(t, hedge_pats, conf_pats)
                            for t in matched["thinking"]], index=matched.index)
        d = matched.copy()
        for col in rec.columns:
            d[col] = rec[col]
        results[name] = {dom: lodo_ap(d, EFFORT_SIGNAL, dom) for dom in DOMAINS}

    print("LODO AP on the judge-matched subset (Table 1 configuration)")
    hdr = f"{'Dictionary':<18}" + "".join(f"{d:>17}" for d in DOMAINS) + f"{'mean':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, per in results.items():
        aps = [per[d] for d in DOMAINS]
        print(f"{name:<18}" + "".join(f"{a:>17.3f}" for a in aps) + f"{np.mean(aps):>9.3f}")

    print(f"\n{'Table 1 (published)':<18}" + "".join(f"{v:>17.2f}" for v in (0.82, 0.92, 0.83, 0.84))
          + f"{0.85:>9.2f}")

    v0 = np.mean([results["V0 deployed"][d] for d in DOMAINS])
    print()
    for name in ("V1 label-blind", "V2 V1+rejected"):
        v = np.mean([results[name][d] for d in DOMAINS])
        print(f"  {name:<18} mean AP delta vs deployed: {v - v0:+.4f}")

if __name__ == "__main__":
    main()
