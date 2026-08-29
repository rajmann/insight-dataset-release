"""Recompute the appendix per-feature Spearman table's MATH column on hard_math_93
(post-cutoff), replacing the MATH-500 Math-Hard/Math-All columns.

9 features (6 effort-signal + 3 trigram-repetition), Spearman rho(feature, correct) with
95% puzzle-clustered bootstrap CI (N=2000, cluster = puzzle_id). Star if CI excludes zero.
Math = hard_math_93, 7 models, scored with _hard_math_score_v2. Features via extract_features_B.

Insight per-domain columns in the existing table are unchanged (computed from features.parquet);
only the math corpus was wrong. This script produces the replacement math numbers.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from train_confidence_classifier import extract_features_B, HEDGE_PATTERNS, count_patterns
from _hard_math_score_v2 import score_row

MROOT = HERE.parent / "data" / "hard_math_93" / "results_un_aug"
SAFE = {"Gemini 2.5 Pro": "Gemini_2_5_Pro", "Gemini 3 Flash": "Gemini_3_Flash",
        "Gemini 3 Pro": "Gemini_3_Pro", "Qwen3-VL-235B Thinking": "Qwen3_VL_235B_Thinking",
        "GPT-5": "GPT_5", "Claude Opus 4.6": "Claude_Opus_4_6", "Claude Sonnet 4.6": "Claude_Sonnet_4_6"}

FEATURES = ["tokens_thinking_proxy", "elapsed", "thinking_char_count",
            "hedge_rate", "hedge_ratio", "hedge_position_variance",
            "bigram_repetition_rate", "trigram_repetition_rate", "unique_trigram_ratio"]


def proxy(raw, thinking):
    r = int(raw or 0)
    return float(r) if r > 0 else len(thinking or "") / 4.0


def load_math():
    rows = []
    for disp, safe in SAFE.items():
        fp = MROOT / f"{safe}.jsonl"
        if not fp.exists():
            continue
        for r in (json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines()):
            th = r.get("thinking") or ""
            if not th.strip():
                continue
            outcome, _ = score_row(safe, r["puzzle_id"], r.get("raw_output") or "", r.get("expected") or "")
            c = {"correct": 1, "wrong": 0}.get(outcome, None)
            if c is None:
                continue
            fb = extract_features_B({"thinking_content": th, "answer": r.get("answer") or "",
                                     "thinking_tokens": int(r.get("tokens_thinking") or 0),
                                     "response_time": float(r.get("elapsed") or 0)})
            row = {"correct": c, "puzzle_id": r["puzzle_id"],
                   "tokens_thinking_proxy": proxy(r.get("tokens_thinking"), th),
                   "elapsed": float(r.get("elapsed") or 0),
                   "thinking_char_count": len(th),
                   "hedge_rate": count_patterns(th, HEDGE_PATTERNS) / max(len(th.split()), 1)}
            for k in ["hedge_ratio", "hedge_position_variance", "bigram_repetition_rate",
                      "trigram_repetition_rate", "unique_trigram_ratio"]:
                row[k] = fb.get(k, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def boot_ci(df, col, N=2000, seed=1):
    d = df.dropna(subset=[col])
    clusters = {k: idx.values for k, idx in d.groupby("puzzle_id").groups.items()}
    keys = np.array(list(clusters))
    rng = np.random.default_rng(seed)
    x_all = d[col].astype(float); y_all = d["correct"].astype(float)
    out = []
    for _ in range(N):
        pick = keys[rng.integers(0, len(keys), len(keys))]
        idx = np.concatenate([clusters[k] for k in pick])
        x = x_all.loc[idx].values; y = y_all.loc[idx].values
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        out.append(spearmanr(x, y).statistic)
    lo, hi = np.percentile(out, 2.5), np.percentile(out, 97.5)
    return lo, hi


def main():
    df = load_math()
    print(f"Math = hard_math_93, n={len(df)}, accuracy {df['correct'].mean():.2f}. "
          f"Spearman rho(feature, correct) + puzzle-clustered bootstrap 95% CI (N=2000).\n")
    print(f"{'feature':26s} | {'rho':>7s} | {'95% CI':>18s} | sig")
    print("-" * 62)
    for f in FEATURES:
        d = df.dropna(subset=[f])
        rho = spearmanr(d[f].astype(float), d["correct"]).statistic
        lo, hi = boot_ci(df, f)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"{f:26s} | {rho:+7.3f} | [{lo:+.3f}, {hi:+.3f}] | {star}")
    print("\n* = 95% CI excludes zero. These replace the Math-Hard/Math-All (MATH-500) columns.")


if __name__ == "__main__":
    main()
