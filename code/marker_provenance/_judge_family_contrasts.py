"""Judge-identity checks with correctness controlled, so no confound is introduced.

Raw score levels cannot be compared across model families: Gemini-family traces
are correct 78% of the time against 60% for the rest, so higher scores on them
are appropriate rather than biased. Two designs avoid that entirely.

  1  Inter-judge difference (ours minus an independent judge) on the SAME trace,
     with correctness and domain controlled. A level shift common to both judges
     cancels; what remains is judge-specific treatment of a family.
  2  Substitution: swap the independent judge's scores into the classifier and
     see whether any published number moves.

Plus discrimination (AUC) by judge and family, which is base-rate invariant.

Ported from the working repository for the public release. Paths resolve through
_paths.py against data/ in this bundle; the analysis itself is unchanged.
"""

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import (PAPER, JUDGE_DIR, AUDIT, EFFORT, DOMAINS, GEMINI_FAMILY,
                    JUDGE_MODEL, SECOND_JUDGE_EXTENDED, selection_puzzles,
                    load_traces_with_features, chosen_conf_table)

HERE = Path(__file__).resolve().parent
GEMINI = {"Gemini 3 Flash", "Gemini 3 Pro", "Gemini 2.5 Pro"}
EFFORT = ["tokens_thinking_proxy", "elapsed", "hedge_position_variance",
          "thinking_char_count", "hedge_rate", "hedge_ratio"]
rng = np.random.default_rng(42)

frames = []
rc = pd.read_csv(JUDGE_DIR / "eureka_judge_full.csv"); rc["pass_type"] = "un_augmented"; frames.append(rc)
vp = pd.read_csv(JUDGE_DIR / "eureka_judge_vp.csv"); frames.append(vp[vp.pass_type == "un_augmented"])
j = pd.concat(frames, ignore_index=True).dropna(subset=["candidate"])
j["puzzle_id"] = j.puzzle_id.astype(str)
j["trace_id"] = j.domain + "|" + j.model + "|" + j.puzzle_id + "|" + j.pass_type
ch = j[(j.is_chosen == 1) & j.judge_confidence.notna()].drop_duplicates("trace_id")
meta = ch.set_index("trace_id")[["model", "domain", "correct", "judge_confidence"]]

rows = []
cand_chosen = {t: set(g.loc[g.is_chosen == 1, "candidate"]) for t, g in j.groupby("trace_id")}
for line in (SECOND_JUDGE_EXTENDED).read_text(encoding="utf-8").splitlines():
    r = json.loads(line); tid = r["trace_id"]
    if tid not in meta.index:
        continue
    for cand, s in (r["scores"] or {}).items():
        if s is None or cand not in cand_chosen.get(tid, set()):
            continue
        m = meta.loc[tid]
        rows.append({"trace_id": tid, "model": m.model, "domain": m.domain,
                     "correct": int(m.correct), "ours": float(m.judge_confidence),
                     "independent": float(s)})
d = pd.DataFrame(rows).drop_duplicates("trace_id")
d["delta"] = d.ours - d.independent
d["own"] = (d.model == JUDGE_MODEL).astype(int)
d["family"] = d.model.isin(GEMINI).astype(int)
print(f"n = {len(d)} traces with both judges on the committed answer "
      f"({int(d.own.sum())} written by the judge's own model, "
      f"{int(d.family.sum())} by the Gemini family)\n")

# ------------------------------------------------------------------ design 1
print("1  Inter-judge difference (ours minus independent), correctness and domain controlled")
X_dom = pd.get_dummies(d.domain, prefix="dom", drop_first=True).astype(float)

def coef_ci(target):
    X = pd.concat([d[[target, "correct"]].astype(float), X_dom], axis=1).values
    y = d.delta.values
    fit = LinearRegression().fit(X, y)
    point = fit.coef_[0]
    boots = []
    for _ in range(4000):
        i = rng.integers(0, len(y), len(y))
        if len(set(X[i, 0])) < 2:
            continue
        boots.append(LinearRegression().fit(X[i], y[i]).coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi

for target, label in (("own", "traces written by the judge's own model"),
                      ("family", "traces written by the judge's model family")):
    p, lo, hi = coef_ci(target)
    sig = "  excludes zero" if (lo > 0 or hi < 0) else "  spans zero"
    print(f"   {label:<44}{p:>+8.2f}   [{lo:+.2f}, {hi:+.2f}]{sig}")

# The two contrasts reported in the paper's judge appendix, both against the four
# non-Gemini models so the comparison group shares no family with the judge.
NON_GEMINI = sorted(set(d.model.unique()) - GEMINI_FAMILY)


def contrast(group, label):
    # reseed so each contrast is reproducible regardless of call order
    rng_local = np.random.default_rng(42)
    sub = d[d.model.isin(set(group) | set(NON_GEMINI))].copy()
    sub["g"] = sub.model.isin(group).astype(float)
    X = pd.concat([sub[["g", "correct"]].astype(float),
                   pd.get_dummies(sub.domain, prefix="dom", drop_first=True).astype(float)],
                  axis=1).values
    y = sub.delta.values
    point = LinearRegression().fit(X, y).coef_[0]
    boots = []
    for _ in range(4000):
        i = rng_local.integers(0, len(y), len(y))
        if len(set(X[i, 0])) == 2:
            boots.append(LinearRegression().fit(X[i], y[i]).coef_[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sig = "  excludes zero" if (lo > 0 or hi < 0) else "  spans zero"
    print(f"   {label:<44}{point:>+8.2f}   [{lo:+.2f}, {hi:+.2f}]{sig}")


print()
print("   Reported in the paper (comparison group: the four non-Gemini models)")
contrast([JUDGE_MODEL], "the judge's own model")
contrast(sorted(GEMINI_FAMILY), "all three Gemini models")
contrast(sorted(GEMINI_FAMILY - {JUDGE_MODEL}), "the two other Gemini models")

# own vs the rest of its own family, the sharpest self-preference test
fam = d[d.family == 1].copy()
Xf = pd.concat([fam[["own", "correct"]].astype(float),
                pd.get_dummies(fam.domain, prefix="dom", drop_first=True).astype(float)], axis=1).values
yf = fam.delta.values
pt = LinearRegression().fit(Xf, yf).coef_[0]
bs = []
for _ in range(4000):
    i = rng.integers(0, len(yf), len(yf))
    if len(set(Xf[i, 0])) == 2:
        bs.append(LinearRegression().fit(Xf[i], yf[i]).coef_[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"   {'own model vs rest of its own family':<44}{pt:>+8.2f}   [{lo:+.2f}, {hi:+.2f}]"
      f"{'  excludes zero' if (lo>0 or hi<0) else '  spans zero'}   (n={len(fam)})")

# ------------------------------------------------------------------ design 2
print("\n2  Substitution: put the independent judge's scores into the classifier")
traces = pd.read_parquet(PAPER / "all_traces.parquet")
traces["puzzle_id"] = traces.puzzle_id.astype(str)
traces["trace_id"] = traces.domain + "|" + traces.model + "|" + traces.puzzle_id + "|" + traces.pass_type
feat = pd.read_parquet(PAPER / "features.parquet")
t = traces[["row_id", "trace_id", "domain", "model", "puzzle_id", "answer", "correct"]].merge(
    feat, on="row_id", how="left")
t["correct"] = t["correct"].astype(int)
t = t[t["answer"].fillna("").str.len() > 0]
sub = t.merge(d[["trace_id", "ours", "independent"]], on="trace_id", how="inner").drop_duplicates("row_id")
print(f"   evaluable on {len(sub)} traces, {sub.domain.nunique()} domains")

def lopo_ap(frame, conf):
    probs = np.zeros(len(frame))
    y = frame.correct.values
    Xe = frame[EFFORT].fillna(0).values
    c = frame[conf].values.reshape(-1, 1)
    for pid in frame.puzzle_id.unique():
        m = (frame.puzzle_id == pid).values
        if y[~m].std() == 0:
            probs[m] = y[~m].mean(); continue
        s = StandardScaler().fit(Xe[~m])
        l1 = LogisticRegression(max_iter=2000).fit(s.transform(Xe[~m]), y[~m])
        p_tr = l1.predict_proba(s.transform(Xe[~m]))[:, 1]
        p_te = l1.predict_proba(s.transform(Xe[m]))[:, 1]
        l2 = LogisticRegression(max_iter=2000).fit(np.column_stack([p_tr, c[~m]]), y[~m])
        probs[m] = l2.predict_proba(np.column_stack([p_te, c[m]]))[:, 1]
    return average_precision_score(y, probs)

for conf, label in (("ours", "our judge"), ("independent", "independent judge")):
    print(f"   stacked effort+conf, LOPO AP with {label:<20}{lopo_ap(sub, conf):.4f}")
