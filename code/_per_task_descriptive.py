"""A4: per-task descriptive statistics (answers yRb7's request for descriptives).

For each task (4 insight domains + hard math): number of puzzles, traces, models,
accuracy, and thinking-token counts (mean +/- std, median). Plus the wrong-vs-correct
token median ratio per task.

Reading (ties descriptives to the paper's arguments):
  W2 (difficulty): within EVERY task, including math, wrong traces run longer (ratio > 1).
      Length/effort tracks struggle - a shared signal, present on math too. So "it's just
      difficulty" is not refuted by length; it is answered by the orthogonal confidence
      axis (R2) and by the trace being the only observable of difficulty at solve time.
  W1 (math contrast): the math wrong/correct ratio (2.5x) sits INSIDE the insight range (1.3-6.8x),
      so length alone does NOT separate math from insight. The contrast lives in the
      LANGUAGE markers (hedge / rejection / eureka), which are flat on math - documented in
      the marker-density analysis, not here. A4 reinforces that length is shared, not distinct.
  Note: token count is not a clean cross-task difficulty proxy - Connections has the most
      tokens AND the lowest accuracy, math has many tokens AND the highest accuracy.

Tokens = tokens_thinking_proxy (canonical: raw thinking tokens where the provider reports
them, else chars//4). Raw tokens_thinking alone is unusable as a descriptive because Claude
and Qwen report 0 (no thinking-token field), so it would measure vendor reporting, not effort.
Insight: all_traces.parquet + features.parquet, un_augmented, answered rows.
Math: hard_math_93/results_un_aug (50 of 93 authored puzzles run un-augmented), scored v2.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _hard_math_score_v2 import score_row

PAPER = HERE.parent / "data" / "insight_4domain"
INSIGHT = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]
MATH_ROOT = HERE.parent / "data" / "hard_math_93" / "results_un_aug"
SAFE = {"Gemini 2.5 Pro": "Gemini_2_5_Pro", "Gemini 3 Flash": "Gemini_3_Flash",
        "Gemini 3 Pro": "Gemini_3_Pro", "Qwen3-VL-235B Thinking": "Qwen3_VL_235B_Thinking",
        "GPT-5": "GPT_5", "Claude Opus 4.6": "Claude_Opus_4_6", "Claude Sonnet 4.6": "Claude_Sonnet_4_6"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def proxy(raw, thinking):
    """Canonical token proxy: raw thinking tokens if reported (>0), else chars//4."""
    r = int(raw or 0)
    return float(r) if r > 0 else len(thinking or "") / 4.0


def collect():
    rows = []
    tr = pd.read_parquet(PAPER / "all_traces.parquet")
    feat = pd.read_parquet(PAPER / "features.parquet")[["row_id", "tokens_thinking_proxy"]]
    un = tr[(tr["pass_type"] == "un_augmented") & (tr["answer"].fillna("").str.len() > 0)]
    un = un.merge(feat, on="row_id", how="left")
    for _, r in un.iterrows():
        rows.append(["insight", r["domain"], r["model"], r["puzzle_id"], int(r["correct"]),
                     fnum(r["tokens_thinking_proxy"]), fnum(r["elapsed_seconds"])])
    for disp, safe in SAFE.items():
        fp = MATH_ROOT / f"{safe}.jsonl"
        if not fp.exists():
            continue
        for r in (json.loads(l) for l in fp.read_text(encoding="utf-8").splitlines()):
            outcome, _ = score_row(safe, r["puzzle_id"], r.get("raw_output") or "", r.get("expected") or "")
            # accuracy over attempted (correct/wrong); no_pred/trunc excluded from acc, kept for count note
            corr = {"correct": 1, "wrong": 0}.get(outcome, None)
            rows.append(["math", "Math", disp, r["puzzle_id"], corr,
                         proxy(r.get("tokens_thinking"), r.get("thinking")),
                         fnum(r.get("elapsed"))])
    return pd.DataFrame(rows, columns=["kind", "task", "model", "puzzle_id", "correct",
                                       "tokens", "elapsed"])


def main():
    df = collect()
    order = INSIGHT + ["Math"]
    print("A4: per-task descriptives. Insight un_augmented (answered rows); math scored v2.")
    print("tokens = tokens_thinking_proxy (raw where reported, else chars//4).")
    print("acc = correct / attempted (excludes no-prediction).\n")
    hdr = (f"{'task':16s} | {'puz':>4s} | {'traces':>6s} | {'mdl':>3s} | {'acc':>5s} | "
           f"{'tok mean':>9s} | {'tok std':>8s} | {'tok med':>8s} | {'wrong/correct':>13s}")
    print(hdr); print("-" * len(hdr))
    for task in order:
        s = df[df["task"] == task]
        att = s[s["correct"].notna()]          # attempted (has correct/wrong label)
        acc = att["correct"].mean() if len(att) else float("nan")
        tok = s["tokens"].dropna()
        w = att[att["correct"] == 0]["tokens"].dropna()
        c = att[att["correct"] == 1]["tokens"].dropna()
        wc = (np.median(w) / np.median(c)) if len(w) and len(c) and np.median(c) else float("nan")
        print(f"{task:16s} | {s['puzzle_id'].nunique():>4d} | {len(s):>6d} | "
              f"{s['model'].nunique():>3d} | {acc:>5.2f} | {tok.mean():>9.0f} | "
              f"{tok.std():>8.0f} | {np.median(tok):>8.0f} | {wc:>12.2f}x")
    print("-" * len(hdr))
    # note dropped (no-pred/trunc) traces for math transparency
    mno = df[(df.kind == "math") & (df.correct.isna())]
    if len(mno):
        print(f"\nMath: {len(mno)} traces excluded from acc (no prediction / truncated).")
    print("\nwrong/correct = median proxy tokens on wrong traces / on correct traces (per task).")
    print(">1 = wrong longer. Present on EVERY task incl. math (2.5x, inside insight's 1.3-6.8x):")
    print("length/effort is a SHARED difficulty signal; the math contrast lives in language")
    print("markers (hedge/rejection/eureka), not in trace length.")


if __name__ == "__main__":
    main()
