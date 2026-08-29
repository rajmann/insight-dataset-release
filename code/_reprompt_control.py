"""Control analysis for the 100% re-prompt sweep: does the classifier's *targeting*
add value, or would re-prompting a random / top quartile do as well?

Slices net accuracy gain (PREFERRED vs pass-1) by classifier-confidence quartile and
builds the escalation curve (re-prompt lowest-confidence-first). Answers the supervisor's
control: gain(bottom quartile, targeted) vs gain(random 25%) vs gain(top quartile).

- "random 25%" net gain == the overall per-trace mean gain (a random subset has the same
  expected per-trace gain), so we report the pooled overall gain as the random baseline.
- Quartiles are on predicted_prob, computed WITHIN (model, domain) then pooled, so a
  low-base-rate model doesn't dominate the bottom bin.

Reuses the paper normaliser + gold via _reprompt_analyse.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _reprompt_analyse import gold_set, is_correct, load_records


def to_frame(recs):
    rows = []
    for r in recs:
        golds = gold_set(r["domain"], r["puzzle_id"], r.get("expected", ""))
        rows.append({
            "model": r["model"], "domain": r["domain"], "puzzle_id": r["puzzle_id"],
            "predicted_prob": float(r["predicted_prob"]),
            "snapshot_sub": bool(r.get("snapshot_note")),
            "p1": int(r["pass1_correct"]),
            "pref": int(is_correct(r.get("preferred_answer", ""), golds)),
        })
    df = pd.DataFrame(rows)
    df["gain"] = df["pref"] - df["p1"]          # +1 recovered, -1 harmed, 0 unchanged
    # confidence quartile within (model, domain): Q1 = lowest confidence (most flagged)
    df["q"] = (df.groupby(["model", "domain"])["predicted_prob"]
                 .transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4])))
    return df


def report(df, label):
    n = len(df)
    if not n:
        print(f"\n[{label}] no rows"); return
    ov = df["gain"].mean()
    print(f"\n===== {label}  (n={n}) =====")
    print(f"  overall net gain (= expected gain of a RANDOM 25%) : {ov*100:+.1f}pp")
    print(f"  {'quartile':<9}{'n':>5}{'pass1':>7}{'pref':>7}"
          f"{'gains':>8}{'losses':>8}{'net':>7}{'rec%':>7}{'harm%':>7}")
    for q in [1, 2, 3, 4]:
        g = df[df["q"] == q]
        if not len(g):
            continue
        p1, pr = g["p1"].mean(), g["pref"].mean()
        rec = int((g.gain == 1).sum()); harm = int((g.gain == -1).sum())
        gains_pp = rec / len(g) * 100; loss_pp = harm / len(g) * 100
        wrong = int((g.p1 == 0).sum()); right = int((g.p1 == 1).sum())
        rec_rate = rec / wrong * 100 if wrong else float("nan")     # of wrong, % recovered
        harm_rate = harm / right * 100 if right else float("nan")   # of right, % broken
        tag = " <-flagged" if q == 1 else (" <-top" if q == 4 else "")
        print(f"  Q{q}{'':<7}{len(g):>5}{p1:>7.3f}{pr:>7.3f}"
              f"{gains_pp:>+8.1f}{-loss_pp:>+8.1f}{(pr-p1)*100:>+7.1f}"
              f"{rec_rate:>6.0f}%{harm_rate:>6.0f}%{tag}")
    # gains / losses / net escalation curves (re-prompt lowest-confidence first)
    s = df.sort_values("predicted_prob")
    rec_cum = (s.gain == 1).cumsum().values      # cumulative recoveries
    harm_cum = (s.gain == -1).cumsum().values    # cumulative harms
    print("  escalation curves (lowest-confidence first; whole-set pp): "
          "gains front-loaded, losses back-loaded")
    print(f"     {'re-prompt':>10}{'gains':>9}{'losses':>9}{'net':>8}")
    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        k = max(1, int(round(frac * n)))
        gpp = rec_cum[k-1] / n * 100; lpp = harm_cum[k-1] / n * 100
        print(f"     {frac*100:>8.0f}%{gpp:>+9.2f}{-lpp:>+9.2f}{(gpp-lpp):>+8.2f}")


def best_threshold(df, label, grid=(5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100)):
    """Net gain on the bottom-k% by confidence — is there a threshold where it turns
    positive, and where is it maximised? (deployment: re-prompt only below the threshold)."""
    n = len(df)
    if not n:
        return
    gains = df.sort_values("predicted_prob")["gain"].values
    best_k, best_g = None, -1e9
    print(f"\n  [{label}] net gain on bottom-k% (deployment threshold search):")
    for kp in grid:
        k = max(1, int(round(kp / 100 * n)))
        g = gains[:k].mean() * 100
        print(f"     bottom {kp:>3}%  (n={k:>3}): {g:+5.1f}pp{'  <- positive' if g > 0 else ''}")
        if g > best_g:
            best_g, best_k = g, kp
    verdict = f"bottom {best_k}% at {best_g:+.1f}pp" if best_g > 0 else \
        f"never net-positive (best bottom {best_k}% still {best_g:+.1f}pp)"
    print(f"     => optimal: {verdict}")


def main():
    df = to_frame(load_records())
    clean = df[~df["snapshot_sub"]]
    report(clean, "POOLED  (same-snapshot models)")
    for dom in ["Rebus", "Cryptic"]:
        report(clean[clean.domain == dom], f"POOLED / {dom}")
    report(df, "POOLED  (all, incl. substituted Gemini 3 Pro)")
    # threshold search: does a narrow low-confidence slice turn net-positive? (esp. Rebus)
    print("\n" + "=" * 60 + "\nDEPLOYMENT THRESHOLD SEARCH\n" + "=" * 60)
    best_threshold(clean, "POOLED")
    for dom in ["Rebus", "Cryptic"]:
        best_threshold(clean[clean.domain == dom], dom)
    print("\nInterpretation: if Q1 net gain >> overall (random) >= Q4, the classifier's "
          "targeting is doing the work, not re-prompting per se.")


if __name__ == "__main__":
    main()
