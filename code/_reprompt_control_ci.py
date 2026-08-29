"""Finalised control stats with puzzle-clustered bootstrap CIs.

Fixes the normaliser inconsistency: scores pass-1 AND preferred with the SAME is_correct,
so net gain is self-consistent (QA: also reports agreement with the paper's stored
pass1_correct label). Then bootstrap CIs (cluster = puzzle_id) on:
  - overall net gain (= random-25% baseline)
  - Q1 (flagged, bottom 25% conf) net gain, Q4 (top) net gain, and Q1 - overall (targeting)
  - per domain: net gain re-prompting everything below a fixed confidence threshold
    (tau10 / tau25 = 10th / 25th pct of predicted_prob), settling Rebus 'rescued' vs 'neutral'.
Pooled over same-snapshot models (excludes substituted Gemini 3 Pro).
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

RNG = np.random.default_rng(0)
NBOOT = 2000


def build():
    rows = []
    for r in load_records():
        golds = gold_set(r["domain"], r["puzzle_id"], r.get("expected", ""))
        p1s = int(is_correct(r["pass1_answer"], golds))          # scored (consistent)
        rows.append({
            "model": r["model"], "domain": r["domain"], "puzzle_id": r["puzzle_id"],
            "pp": float(r["predicted_prob"]), "sub": bool(r.get("snapshot_note")),
            "p1_scored": p1s, "p1_stored": int(r["pass1_correct"]),
            "pref": int(is_correct(r.get("preferred_answer", ""), golds)),
        })
    df = pd.DataFrame(rows)
    df["gain"] = df["pref"] - df["p1_scored"]
    df["q"] = (df.groupby(["model", "domain"])["pp"]
                 .transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4])).astype(int))
    return df


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def boot(df, fn):
    puz = df["puzzle_id"].unique()
    idx = {p: df.index[df.puzzle_id == p].to_numpy() for p in puz}
    out = []
    for _ in range(NBOOT):
        pick = RNG.choice(puz, len(puz), replace=True)
        rows = np.concatenate([idx[p] for p in pick])
        v = fn(df.loc[rows])
        if v is not None and not np.isnan(v):
            out.append(v)
    return np.mean(out), *ci(out)


def line(name, df, fn):
    pt = fn(df)
    m, lo, hi = boot(df, fn)
    star = "*" if (lo > 0 or hi < 0) else " "
    print(f"  {name:<34} {pt*100:+6.1f}pp   95% CI [{lo*100:+5.1f}, {hi*100:+5.1f}] {star}")


def main():
    df = build()
    agree = (df.p1_scored == df.p1_stored).mean()
    print(f"QA: scored-vs-stored pass1 agreement = {agree:.3f} "
          f"({(df.p1_scored != df.p1_stored).sum()} / {len(df)} disagree)\n")
    d = df[~df["sub"]]                                # same-snapshot pool
    print(f"===== POOLED same-snapshot (n={len(d)}); puzzle-clustered bootstrap N={NBOOT} =====")
    line("overall net gain (RANDOM 25%)", d, lambda x: x.gain.mean())
    line("Q1 flagged (bottom 25% conf)", d, lambda x: x[x.q == 1].gain.mean())
    line("Q4 top (most confident)", d, lambda x: x[x.q == 4].gain.mean())
    line("targeting advantage  Q1 - overall", d,
         lambda x: x[x.q == 1].gain.mean() - x.gain.mean())
    for dom in ["Rebus", "Cryptic"]:
        g = d[d.domain == dom]
        t10 = g.pp.quantile(0.10); t25 = g.pp.quantile(0.25)
        print(f"\n  -- {dom} (n={len(g)}) --")
        line(f"{dom} overall", g, lambda x: x.gain.mean())
        line(f"{dom} re-prompt <= tau25 (bottom 25%)", g, lambda x, t=t25: x[x.pp <= t].gain.mean())
        line(f"{dom} re-prompt <= tau10 (bottom 10%)", g, lambda x, t=t10: x[x.pp <= t].gain.mean())
    print("\n * = 95% CI excludes zero.")


if __name__ == "__main__":
    main()
