"""Stage 3 — fit LR across the (config × feature_subset × cv_setup × domain) grid.

Produces predictions.parquet in long format:
  row_id, config, feature_subset, cv_setup, fold, predicted_prob, true_label

This is the input to Stage 4 (metrics) and Stage 5 (cascade).

Configs:
  (a) un_augmented trace, no SR feature           — single-call deployable
  (b) augmented trace, no SR feature               — single-call with SR ask, classifier ignores SR
  (c) augmented trace, with SR feature             — single-call deployable + SR     [DEPLOYABLE BASELINE]
  (d) un_augmented trace, SR from augmented row    — theoretical two-call hybrid (v1 default)

Feature subsets (some are config-conditional):
  canonical_25 / canonical_23  — all features (with/without SR per config)
  effort_only                  — token counts + elapsed + SR (5 features when SR present, 3 without)
  trajectory_only              — 4 trajectory features
  text_pattern_only            — 13 text-pattern features + 2 length

CV setups:
  lopo_within_domain     — leave-one-puzzle-out within each domain
  fold5_domain_stratified — 5-fold, balanced by domain (matches Phase 2 setup)
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
OUT_DIR = HERE / os.environ.get("PAPER_DATASET", "paper_dataset_2026-04-27")
TRACES = OUT_DIR / "all_traces.parquet"
FEATURES = OUT_DIR / "features.parquet"
PREDICTIONS = OUT_DIR / "predictions.parquet"


# Feature schema. tokens_thinking_proxy is the canonical token-effort feature
# (provider-unified; see _tokens_proxy_test.compute_proxy). The raw
# tokens_thinking and tokens_output columns are present in features.parquet
# for backward compat / ablation testing but are NOT in canonical subsets.
SR_FEATURES = ["self_reported_conf", "self_reported_present"]
EFFORT_TOKENS = ["tokens_thinking_proxy"]                  # canonical: proxy only
EFFORT_TOKENS_RAW = ["tokens_thinking", "tokens_output"]   # v1-equivalent for cross-check
EFFORT_NON_TOKEN = ["elapsed"]                              # always present
TRAJECTORY = ["n_candidate_mentions", "pred_is_last_candidate",
              "pred_mention_count", "candidate_churn"]
TEXT_PATTERN = ["hedge_count", "confidence_count", "explicit_rejection_count",
                "hedge_ratio", "hedge_shift", "hedge_density_last_third",
                "hedge_position_variance", "candidate_switch_count",
                "revision_count", "self_correction_count",
                "question_marks", "exclamation_marks",
                "bigram_repetition_rate", "trigram_repetition_rate",
                "unique_trigram_ratio"]
LENGTH = ["thinking_word_count", "thinking_char_count"]

# Per-word rate features (length-independent), added 2026-05-05.
# Replaces the 8 raw-count features in the LEN+RATES canonical subsets.
TEXT_PATTERN_RATES = ["hedge_rate", "confidence_rate", "explicit_rejection_rate",
                      "candidate_switch_rate", "revision_rate", "self_correction_rate",
                      "question_mark_rate", "exclamation_mark_rate"]
TEXT_NORMALISED = ["hedge_ratio", "hedge_shift", "hedge_density_last_third",
                   "hedge_position_variance",
                   "bigram_repetition_rate", "trigram_repetition_rate",
                   "unique_trigram_ratio"]
# Non-leaking trajectory subset (drops pred_mention_count + pred_is_last_candidate
# which degenerate to 0 on rows where the model did not commit to an answer).
TRAJECTORY_NO_LEAKS = ["n_candidate_mentions", "candidate_churn"]

# Canonical feature compositions:
#   canonical:           proxy + elapsed + (SR) + traj + text + length  → §R-aligned, with SR-feature option
#   canonical_no_tokens: drop the token-effort feature entirely (ablation)
#   canonical_v1raw:     v1-style raw tokens (matches §R.0.1 PRE-drop schema, for cross-check)
CANONICAL_NO_SR = EFFORT_TOKENS + EFFORT_NON_TOKEN + TRAJECTORY + TEXT_PATTERN + LENGTH
CANONICAL_WITH_SR = SR_FEATURES + CANONICAL_NO_SR
CANONICAL_NO_TOKENS_NO_SR = EFFORT_NON_TOKEN + TRAJECTORY + TEXT_PATTERN + LENGTH
CANONICAL_NO_TOKENS_WITH_SR = SR_FEATURES + CANONICAL_NO_TOKENS_NO_SR
CANONICAL_V1RAW_NO_SR = EFFORT_TOKENS_RAW + EFFORT_NON_TOKEN + TRAJECTORY + TEXT_PATTERN + LENGTH
CANONICAL_V1RAW_WITH_SR = SR_FEATURES + CANONICAL_V1RAW_NO_SR

# LEN + RATES — new canonical from 2026-05-05 ablation. Replaces raw count
# features with per-word rates and drops the trivially-leaking trajectory
# features (pred_mention_count, pred_is_last_candidate). Beats LEN+COUNTS by
# +0.011 to +0.026 AUC across 4 domains; ALL adds nothing over LEN+RATES.
CANONICAL_LEN_RATES_NO_SR = (EFFORT_TOKENS + EFFORT_NON_TOKEN +
                              TRAJECTORY_NO_LEAKS +
                              TEXT_PATTERN_RATES + TEXT_NORMALISED + LENGTH)
CANONICAL_LEN_RATES_WITH_SR = SR_FEATURES + CANONICAL_LEN_RATES_NO_SR

FEATURE_SUBSETS_NO_SR = {
    "canonical_no_sr": CANONICAL_NO_SR,
    "canonical_no_tokens_no_sr": CANONICAL_NO_TOKENS_NO_SR,
    "canonical_v1raw_no_sr": CANONICAL_V1RAW_NO_SR,
    "canonical_len_rates_no_sr": CANONICAL_LEN_RATES_NO_SR,  # new 2026-05-05
    "effort_only_no_sr": EFFORT_TOKENS + EFFORT_NON_TOKEN,
    "effort_no_tokens_no_sr": EFFORT_NON_TOKEN,
    "trajectory_only": TRAJECTORY,
    "text_pattern_only": TEXT_PATTERN + LENGTH,
    "rates_only_no_sr": TEXT_PATTERN_RATES + TEXT_NORMALISED,  # new 2026-05-05
}
FEATURE_SUBSETS_WITH_SR = {
    "canonical_with_sr": CANONICAL_WITH_SR,
    "canonical_no_tokens_with_sr": CANONICAL_NO_TOKENS_WITH_SR,
    "canonical_v1raw_with_sr": CANONICAL_V1RAW_WITH_SR,
    "canonical_len_rates_with_sr": CANONICAL_LEN_RATES_WITH_SR,  # new 2026-05-05
    "effort_only_with_sr": EFFORT_TOKENS + EFFORT_NON_TOKEN + SR_FEATURES,
    "effort_no_tokens_with_sr": EFFORT_NON_TOKEN + SR_FEATURES,
    "trajectory_only": TRAJECTORY,
    "text_pattern_only": TEXT_PATTERN + LENGTH,
    "sr_only": SR_FEATURES,
}


# ── Build per-config row sets ───────────────────────────────────────────────

def build_config_rows(traces, features):
    """Return dict {config_name: dataframe-of-rows-with-features-and-label}.

    Each per-config dataframe has columns:
      row_id, domain, model, puzzle_id, correct, <feature columns>
    For config (d), we replace SR features in the un_augmented row with
    SR values cross-joined from the same (domain, model, puzzle_id)
    augmented row.
    """
    # The feature extractor already produces normalised versions of token/elapsed/SR;
    # drop the raw columns from traces before merging to avoid name collisions.
    drop_cols = [c for c in ["self_reported_conf", "tokens_thinking", "tokens_output"]
                 if c in traces.columns]
    traces_keep = traces.drop(columns=drop_cols)
    df = traces_keep.merge(features, on="row_id", how="left")
    # Fill any NaN feature values with 0 (e.g. partial backfill rows with empty traces)
    all_feat_cols = list(set(CANONICAL_WITH_SR + CANONICAL_V1RAW_WITH_SR
                              + CANONICAL_NO_TOKENS_WITH_SR
                              + ["_trajectory_from_un_aug"]))
    feat_cols = [c for c in df.columns if c in all_feat_cols]
    df[feat_cols] = df[feat_cols].fillna(0)
    out = {}

    # Config (a): un_augmented, no SR feature
    a = df[df["pass_type"] == "un_augmented"].copy()
    out["a"] = a

    # Config (b): augmented, no SR feature
    b = df[df["pass_type"] == "augmented"].copy()
    out["b"] = b

    # Config (c): augmented, with SR feature (same rows as b — feature subset
    # decides whether SR is used)
    out["c"] = b

    # Config (d): un_augmented, SR cross-joined from augmented row
    aug_sr = df[df["pass_type"] == "augmented"][
        ["domain", "model", "puzzle_id"] + SR_FEATURES
    ].rename(columns={
        "self_reported_conf": "_aug_sr_conf",
        "self_reported_present": "_aug_sr_present",
    })
    d = a.merge(aug_sr, on=["domain", "model", "puzzle_id"], how="left")
    # Override the un_augmented row's SR (which was 0.5/0) with augmented SR
    # where available; leave 0.5/0 fallback otherwise.
    d["self_reported_conf"] = d["_aug_sr_conf"].where(d["_aug_sr_conf"].notna(),
                                                        d["self_reported_conf"])
    d["self_reported_present"] = d["_aug_sr_present"].where(d["_aug_sr_present"].notna(),
                                                              d["self_reported_present"])
    out["d"] = d.drop(columns=["_aug_sr_conf", "_aug_sr_present"])

    return out


# ── CV splitters ────────────────────────────────────────────────────────────

def cv_lopo_within_domain(df):
    """Yield (fold_id, train_mask, test_mask) per held-out puzzle_id within each domain."""
    fold = 0
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        for pid in sorted(sub["puzzle_id"].unique()):
            test_mask = (df["domain"] == domain) & (df["puzzle_id"] == pid)
            train_mask = (df["domain"] == domain) & ~test_mask & (df["domain"] == domain)
            # Restrict train to same domain
            yield fold, (df["domain"] == domain) & ~((df["domain"] == domain) & (df["puzzle_id"] == pid)), test_mask
            fold += 1


def cv_fold5_domain_stratified(df, seed=42):
    """5-fold split, with each fold roughly balanced across domains."""
    fold_assignment = np.full(len(df), -1, dtype=int)
    for domain in df["domain"].unique():
        idx = np.where(df["domain"].values == domain)[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_id, (_, te) in enumerate(kf.split(idx)):
            fold_assignment[idx[te]] = fold_id
    for fold_id in range(5):
        test_mask = (fold_assignment == fold_id)
        train_mask = ~test_mask & (fold_assignment >= 0)
        yield fold_id, pd.Series(train_mask, index=df.index), pd.Series(test_mask, index=df.index)


# ── LR fit + predict ────────────────────────────────────────────────────────

def fit_predict(X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    Xtr = sc.transform(X_tr)
    Xte = sc.transform(X_te)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xte)[:, 1]


def run_sweep(traces_path=TRACES, features_path=FEATURES, out_path=PREDICTIONS,
              configs=("a", "b", "c", "d"),
              cv_setups=("lopo_within_domain", "fold5_domain_stratified")):
    print("Loading traces + features...")
    traces = pd.read_parquet(traces_path)
    features = pd.read_parquet(features_path)
    print(f"  traces: {len(traces)}  features: {len(features)}")

    print("\nBuilding per-config row sets...")
    config_dfs = build_config_rows(traces, features)
    for c, df in config_dfs.items():
        print(f"  ({c}): {len(df)} rows")

    predictions = []

    for config in configs:
        df = config_dfs[config].reset_index(drop=True)
        if df.empty:
            continue
        # Pick subsets per config
        subsets = (FEATURE_SUBSETS_WITH_SR if config in ("c", "d")
                   else FEATURE_SUBSETS_NO_SR)

        for fs_name, fs_cols in subsets.items():
            # Skip if any column is missing (some are config-conditional)
            missing = [c for c in fs_cols if c not in df.columns]
            if missing:
                print(f"  ({config}, {fs_name}): skip — missing {missing}")
                continue

            X = df[fs_cols].values.astype(float)
            y = df["correct"].values.astype(int)

            for cv_name in cv_setups:
                cv_iter = (cv_lopo_within_domain if cv_name == "lopo_within_domain"
                           else cv_fold5_domain_stratified)
                preds = np.full(len(df), np.nan)
                folds = np.full(len(df), -1, dtype=int)
                for fold_id, tr_mask, te_mask in cv_iter(df):
                    tr_mask = tr_mask.values if hasattr(tr_mask, "values") else tr_mask
                    te_mask = te_mask.values if hasattr(te_mask, "values") else te_mask
                    if tr_mask.sum() < 5 or te_mask.sum() == 0:
                        continue
                    p = fit_predict(X[tr_mask], y[tr_mask], X[te_mask])
                    preds[te_mask] = p
                    folds[te_mask] = fold_id

                done = (~np.isnan(preds)).sum()
                print(f"  ({config}, {fs_name}, {cv_name}): {done}/{len(df)} predictions")
                for i in range(len(df)):
                    if np.isnan(preds[i]):
                        continue
                    predictions.append({
                        "row_id": df.iloc[i]["row_id"],
                        "config": config,
                        "feature_subset": fs_name,
                        "cv_setup": cv_name,
                        "fold": int(folds[i]),
                        "predicted_prob": float(preds[i]),
                        "true_label": int(y[i]),
                        "domain": df.iloc[i]["domain"],
                        "model": df.iloc[i]["model"],
                        "puzzle_id": df.iloc[i]["puzzle_id"],
                    })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({len(pred_df)} rows, "
          f"{out_path.stat().st_size/1024/1024:.2f} MB)")
    print(f"  unique configs: {pred_df['config'].unique().tolist()}")
    print(f"  unique feature_subsets: {pred_df['feature_subset'].unique().tolist()}")
    print(f"  unique cv_setups: {pred_df['cv_setup'].unique().tolist()}")
    return pred_df


def main():
    run_sweep()


if __name__ == "__main__":
    main()
