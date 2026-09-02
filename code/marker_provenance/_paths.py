"""Shared paths for the marker-provenance and judge-validation analyses.

These scripts were written against the working repository and are re-pointed
here at the released layout, so the numbers in the paper's marker-provenance
and judge appendices can be reproduced from this bundle alone.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PAPER = ROOT / "data" / "insight_4domain"
JUDGE_DIR = ROOT / "data" / "judge_scores"
AUDIT = ROOT / "code" / "audit_step2" / "data"

EUREKA_FULL = JUDGE_DIR / "eureka_judge_full.csv"      # Rebus + Cryptic, per candidate
EUREKA_VP = JUDGE_DIR / "eureka_judge_vp.csv"          # VisualPuzzles, per candidate
EUREKA_CONNP = JUDGE_DIR / "eureka_judge_connp.csv"    # Connections+, per group
SECOND_JUDGE_EXTENDED = JUDGE_DIR / "second_judge_extended.jsonl"

EFFORT = ["tokens_thinking_proxy", "elapsed", "hedge_position_variance",
          "thinking_char_count", "hedge_rate", "hedge_ratio"]
DOMAINS = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]
GEMINI_FAMILY = {"Gemini 3 Flash", "Gemini 3 Pro", "Gemini 2.5 Pro"}
JUDGE_MODEL = "Gemini 3 Flash"


def selection_puzzles() -> set[str]:
    """The 20 Rebus puzzles the marker lists were drawn from."""
    d = json.loads((JUDGE_DIR / "marker_selection_puzzles.json").read_text(encoding="utf-8"))
    return set(d["puzzle_ids"])


def load_traces_with_features():
    """all_traces joined to features, extracted rows only (the analysis frame)."""
    import pandas as pd
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    traces["puzzle_id"] = traces.puzzle_id.astype(str)
    feat = pd.read_parquet(PAPER / "features.parquet")
    cols = ["row_id", "domain", "model", "puzzle_id", "answer", "correct", "thinking"]
    df = traces[cols].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    return df[df["answer"].fillna("").str.len() > 0].reset_index(drop=True)


def chosen_conf_table():
    """One row per (domain, model, puzzle_id) with the judge's committed-answer score.

    This is the `chosen_conf` feature used by the stacked classifier. Table 2 and
    the marker appendix are computed on the rows where it exists.
    """
    import pandas as pd
    rc = pd.read_csv(EUREKA_FULL)
    rc = rc[rc.is_chosen == 1].dropna(subset=["judge_confidence"]).copy()
    rc["chosen_conf"] = rc.judge_confidence

    vp = pd.read_csv(EUREKA_VP)
    vp = vp[(vp.is_chosen == 1) & (vp.pass_type == "un_augmented")].dropna(
        subset=["judge_confidence"]).copy()
    vp["chosen_conf"] = vp.judge_confidence

    cp = pd.read_csv(EUREKA_CONNP)
    cp = cp[cp.error.isna()].copy()

    def mean_agg(gs):
        if pd.isna(gs):
            return None
        try:
            vals = [s for s in json.loads(gs) if isinstance(s, (int, float))]
        except Exception:
            return None
        return sum(vals) / len(vals) if vals else None

    cp["chosen_conf"] = [mean_agg(g) for g in cp.group_scores]
    cp["domain"] = "ConnectionsPlus"
    cp = cp.dropna(subset=["chosen_conf"])

    cols = ["domain", "model", "puzzle_id", "chosen_conf"]
    out = pd.concat([rc[cols], vp[cols], cp[cols]], ignore_index=True)
    out["puzzle_id"] = out.puzzle_id.astype(str)
    return out
