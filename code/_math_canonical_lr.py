"""Canonical-feature LR on math un_aug — within-domain LOPO.

Reuses the existing `extract_features_B` text-pattern extractor and adds
effort + math-specific trajectory features (number of times the model's
predicted answer appears in the thinking, position of first/last mention).

Caveats for math vs the other 3 domains:
  - Base rate is high (~92% correct), so AP no-skill baseline is ~0.08.
    Within-domain AP is harder to interpret cross-domain.
  - tokens_thinking is 0 for OpenRouter providers (Claude / Qwen / GPT-5);
    we use the same per-provider proxy logic as elsewhere (char-count fallback).
  - Self-report is 0 for un_aug (no confidence ask in the prompt).

Outputs:
  - LOPO AP per model (and pooled)
  - LOPO AUC per model (and pooled)
  - Top features by |coefficient| in pooled fit
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add llm_evaluation/ to import path so we can reuse extract_features_B
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # llm_evaluation/
sys.path.insert(0, str(ROOT))

from train_confidence_classifier import extract_features_B  # noqa: E402

DATA_DIR = HERE / "math_100" / "results_un_aug"

MODELS = [
    ("Gemini 3 Pro",              "Gemini_3_Pro"),
    ("Gemini 3 Flash",            "Gemini_3_Flash"),
    ("Gemini 2.5 Pro",            "Gemini_2_5_Pro"),
    ("GPT-5",                     "GPT_5"),
    ("Claude Sonnet 4.6",         "Claude_Sonnet_4_6"),
    ("Claude Opus 4.6",           "Claude_Opus_4_6"),
    ("Qwen3-VL-235B Thinking",    "Qwen3_VL_235B_Thinking"),
]


def normalise_math(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    m = re.match(r"^\\boxed\{(.*)\}$", s)
    if m:
        s = m.group(1)
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\left|\\right|\\,|\\!|\\;|\\:|\\text\{[^{}]+\}", "", s)
    s = re.sub(r"\^\\?circ|\^{\\?circ}", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def is_correct(answer, expected):
    if answer is None:
        return False
    a, e = normalise_math(answer), normalise_math(expected)
    if not a or not e:
        return False
    if a == e:
        return True

    def to_float(x):
        try:
            if "/" in x:
                num, den = x.split("/", 1)
                return float(num) / float(den)
            return float(x)
        except (ValueError, ZeroDivisionError):
            return None
    fa, fe = to_float(a), to_float(e)
    if fa is not None and fe is not None and abs(fa - fe) < 1e-9:
        return True
    return False


def math_trajectory_features(thinking, predicted_answer):
    """Math analog of n_candidate_mentions / pred_mention_count / pred_is_last_candidate.

    Counts mentions of the predicted answer in the thinking trace, and locates
    its first/last positions. Mirrors the rebus pred_mention_count logic.
    """
    if not predicted_answer:
        return {
            "pred_mention_count": 0,
            "pred_is_last_candidate": 0,
            "answer_first_pos_frac": 1.0,
            "answer_last_pos_frac": 1.0,
        }
    norm_pred = normalise_math(predicted_answer)
    norm_think = normalise_math(thinking)
    if not norm_pred or len(norm_pred) < 1:
        return {
            "pred_mention_count": 0,
            "pred_is_last_candidate": 0,
            "answer_first_pos_frac": 1.0,
            "answer_last_pos_frac": 1.0,
        }
    count = norm_think.count(norm_pred)
    if count == 0:
        return {
            "pred_mention_count": 0,
            "pred_is_last_candidate": 0,
            "answer_first_pos_frac": 1.0,
            "answer_last_pos_frac": 1.0,
        }
    first = norm_think.find(norm_pred) / max(len(norm_think), 1)
    last = norm_think.rfind(norm_pred) / max(len(norm_think), 1)
    # "Pred is last candidate" — proxy: predicted answer is mentioned in last 20%
    is_last = int(last >= 0.8)
    return {
        "pred_mention_count": count,
        "pred_is_last_candidate": is_last,
        "answer_first_pos_frac": first,
        "answer_last_pos_frac": last,
    }


def tokens_thinking_proxy(rec):
    """Proxy for cross-provider thinking-token count.

    Gemini API reports tokens_thinking. OpenRouter (Claude/Qwen/GPT-5) does not,
    so use char_count // 4 as a rough estimate (matches the canonical pipeline).
    """
    raw = int(rec.get("tokens_thinking") or 0)
    if raw > 0:
        return raw
    chars = len(rec.get("thinking") or "")
    return chars // 4


def load_rows():
    rows = []
    for model_name, fname in MODELS:
        path = DATA_DIR / f"{fname}.jsonl"
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            ans = r.get("answer_reparsed") or r.get("answer")
            thinking = r.get("thinking") or ""
            # OpenRouter Claude/Qwen often have empty thinking; fall back to raw_output
            # so text features have something to operate on.
            if not thinking:
                thinking = r.get("raw_output") or ""
            rec = {
                "model": model_name,
                "puzzle_id": r["puzzle_id"],
                "subject": r.get("subject", ""),
                "level": r.get("level"),
                "expected": r.get("expected", ""),
                "answer": ans,
                "elapsed": float(r.get("elapsed", 0) or 0),
                "tokens_thinking": int(r.get("tokens_thinking", 0) or 0),
                "tokens_output": int(r.get("tokens_output", 0) or 0),
                "thinking": thinking,
                "correct": int(is_correct(ans, r.get("expected", ""))),
            }
            # Effort + proxy
            rec["tokens_thinking_proxy"] = tokens_thinking_proxy(rec)

            # Text-pattern features via reused extractor
            v1 = {
                "thinking_content": thinking,
                "answer": ans or "",
                "thinking_tokens": rec["tokens_thinking"],
                "response_time": rec["elapsed"],
            }
            feats_text = extract_features_B(v1)

            # Math-specific trajectory
            feats_traj = math_trajectory_features(thinking, ans)

            rec.update({
                "hedge_count": feats_text.get("hedge_count", 0),
                "confidence_count": feats_text.get("confidence_count", 0),
                "hedge_ratio": feats_text.get("hedge_ratio", 0.0),
                "hedge_shift": feats_text.get("hedge_shift", 1.0),
                "hedge_density_last_third": feats_text.get("hedge_density_last_third", 0.0),
                "hedge_position_variance": feats_text.get("hedge_position_variance", 0.0),
                "candidate_switch_count": feats_text.get("candidate_switch_count", 0),
                "revision_count": feats_text.get("revision_count", 0),
                "self_correction_count": feats_text.get("self_correction_count", 0),
                "explicit_rejection_count": feats_text.get("explicit_rejection_count", 0),
                "question_marks": feats_text.get("question_marks", 0),
                "exclamation_marks": feats_text.get("exclamation_marks", 0),
                "bigram_repetition_rate": feats_text.get("bigram_repetition_rate", 0.0),
                "trigram_repetition_rate": feats_text.get("trigram_repetition_rate", 0.0),
                "unique_trigram_ratio": feats_text.get("unique_trigram_ratio", 0.0),
                "thinking_word_count": feats_text.get("thinking_word_count", 0),
                "thinking_char_count": feats_text.get("thinking_char_count", 0),
            })
            rec.update(feats_traj)
            rows.append(rec)
    return pd.DataFrame(rows)


# Canonical-25 minus self-report features (un_aug pass has no SR)
FEATURE_COLS = [
    # Effort
    "tokens_thinking_proxy", "tokens_output", "elapsed",
    # Trajectory (math-specific)
    "pred_mention_count", "pred_is_last_candidate",
    "answer_first_pos_frac", "answer_last_pos_frac",
    # Text-pattern
    "hedge_count", "confidence_count", "hedge_ratio", "hedge_shift",
    "hedge_density_last_third", "hedge_position_variance",
    "candidate_switch_count", "revision_count", "self_correction_count",
    "explicit_rejection_count", "question_marks", "exclamation_marks",
    "bigram_repetition_rate", "trigram_repetition_rate", "unique_trigram_ratio",
    # Length
    "thinking_word_count", "thinking_char_count",
]


def lopo_eval(df, target_model=None):
    """Leave-one-puzzle-out CV. If target_model given, restrict eval to that model."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Within-domain pool: train on all rows except those with the held-out puzzle,
    # predict on the held-out puzzle's rows.
    puzzle_ids = sorted(df["puzzle_id"].unique())
    probs = np.zeros(len(df))
    feats = df[FEATURE_COLS].fillna(0).values
    y = df["correct"].values  # NOTE: predicting correct=1, so AP measures "rank correct above wrong".
    # We want to predict WRONG (the rare class). Flip target.
    y_wrong = 1 - y

    for pid in puzzle_ids:
        mask = (df["puzzle_id"] == pid).values
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(feats[~mask])
        Xte = scaler.transform(feats[mask])
        ytr = y_wrong[~mask]
        if ytr.sum() < 2:
            # Fall back to base-rate prediction if no wrong-class examples in train
            probs[mask] = ytr.mean()
            continue
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Xtr, ytr)
        probs[mask] = lr.predict_proba(Xte)[:, 1]

    if target_model is not None:
        m_mask = (df["model"] == target_model).values
        if m_mask.sum() == 0 or y_wrong[m_mask].sum() == 0:
            return None, None, m_mask.sum()
        ap = average_precision_score(y_wrong[m_mask], probs[m_mask])
        try:
            auc = roc_auc_score(y_wrong[m_mask], probs[m_mask])
        except ValueError:
            auc = float("nan")
        return ap, auc, m_mask.sum()

    ap = average_precision_score(y_wrong, probs)
    auc = roc_auc_score(y_wrong, probs)
    return ap, auc, len(df)


def main():
    print("Loading and feature-extracting math un_aug rows...")
    df = load_rows()
    print(f"  {len(df)} rows loaded")
    print(f"  base rate (correct): {df['correct'].mean():.3f}")
    print(f"  base rate (wrong):   {1 - df['correct'].mean():.3f}")
    print()

    print("Per-model accuracy:")
    for model_name, _ in MODELS:
        sub = df[df["model"] == model_name]
        n = len(sub)
        acc = sub["correct"].mean()
        n_wrong = int((1 - sub["correct"]).sum())
        print(f"  {model_name:<28s} n={n:3d}  acc={acc:.3f}  n_wrong={n_wrong}")
    print()

    print("Pooled within-domain LOPO (target = wrong, since wrong is the minority class):")
    ap, auc, n = lopo_eval(df)
    base = 1 - df["correct"].mean()
    lift_ap = ap - base  # lift over no-skill AP baseline
    print(f"  pooled (n={n}): AP={ap:.3f} [no-skill={base:.3f}, lift={lift_ap:+.3f}]  AUC={auc:.3f}")
    print()

    print("Per-model LOPO evaluation (training on pooled, scoring per model):")
    print(f"  {'Model':<28s} {'n':>5s} {'n_wrong':>8s} {'base':>7s} {'AP':>7s} {'lift':>7s} {'AUC':>7s}")
    for model_name, _ in MODELS:
        sub = df[df["model"] == model_name]
        n_wrong = int((1 - sub["correct"]).sum())
        if n_wrong < 2:
            print(f"  {model_name:<28s} {len(sub):>5d} {n_wrong:>8d}   skipped (n_wrong<2)")
            continue
        ap, auc, n = lopo_eval(df, target_model=model_name)
        base = 1 - sub["correct"].mean()
        lift = (ap - base) if ap is not None else float("nan")
        print(f"  {model_name:<28s} {n:>5d} {n_wrong:>8d} {base:>7.3f} {ap:>7.3f} {lift:>+7.3f} {auc:>7.3f}")

    # Top features by |coef| from a pooled fit on the full data
    print()
    print("Top 12 features by |coefficient| in pooled fit (no CV, on full data):")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    y_wrong = (1 - df["correct"]).values
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X, y_wrong)
    coef_abs = sorted(zip(FEATURE_COLS, lr.coef_[0]), key=lambda x: -abs(x[1]))[:12]
    for name, c in coef_abs:
        sign = "+" if c > 0 else "-"
        print(f"  {name:<32s} {sign} {abs(c):.3f}")


if __name__ == "__main__":
    main()
