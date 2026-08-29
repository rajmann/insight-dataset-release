"""Cascade routing value via area-above-random, pooled + per-domain.

The original cascade test (cascade_4domain.json) compared classifier vs random
escalation at a SINGLE operating point per domain (n=100), and found no
significant advantage. That test is stringent and low-powered.

This re-analysis uses the whole cost-accuracy curve. Random escalation at rate
r has EXACT expected accuracy = (1-r)*A_flash + r*A_pro (a straight line from
Solo-Flash to Solo-Pro), because escalating a random fraction r gives each
puzzle its stage-2 outcome with prob r. The classifier's routing value is how
far its accuracy-vs-escalation curve bows ABOVE that line, integrated over the
sweep. We bootstrap the area (clustered by puzzle) per domain and pooled.

Metric: mean over escalation rate of [classifier_acc(r) - random_line(r)], in
accuracy points. Also max gap and the escalation rate needed to reach Solo-Pro
accuracy (classifier) vs random (which only reaches it at r=1).

No new API calls: reuses predictions.parquet + all_traces.parquet.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "data" / "insight_4domain"
N_BOOT = 2000
SEED = 42
COST = {"Gemini 3 Flash": 0.00080, "Gemini 3 Pro": 0.0339}
M1, M2 = "Gemini 3 Flash", "Gemini 3 Pro"
DOMAINS = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]


def curve_area(s1c, s2c, sc):
    """Mean, over the full escalation sweep, of (classifier_acc - random_line).
    Returns area (fraction), max_gap, esc_at_max, esc_to_match_pro."""
    n = len(s1c)
    a_flash, a_pro = s1c.mean(), s2c.mean()
    order = np.argsort(sc)               # lowest predicted-correct first -> escalate first
    s1o, s2o = s1c[order], s2c[order]
    # cumulative: escalate first k -> those get s2, rest keep s1
    esc_correct = np.concatenate([[0], np.cumsum(s2o)])          # sum s2 over first k
    keep_correct_total = s1c.sum()
    keep_correct = keep_correct_total - np.concatenate([[0], np.cumsum(s1o)])  # s1 over remaining
    k = np.arange(n + 1)
    clf_acc = (esc_correct + keep_correct) / n
    r = k / n
    line = (1 - r) * a_flash + r * a_pro
    gap = clf_acc - line
    area = gap.mean()
    imax = int(np.argmax(gap))
    # escalation needed for classifier to reach solo-pro accuracy
    reach = np.where(clf_acc >= a_pro - 1e-9)[0]
    esc_to_pro = float(reach[0] / n) if len(reach) else 1.0
    return area, float(gap[imax]), float(r[imax]), esc_to_pro, a_flash, a_pro


def load_domain(p_sub, correct, dom):
    rows = p_sub[(p_sub["domain"] == dom) & (p_sub["model"] == M1)]
    s1c, s2c, sc, pids = [], [], [], []
    for _, rr in rows.iterrows():
        a = correct.get((M1, rr["puzzle_id"], dom)); b = correct.get((M2, rr["puzzle_id"], dom))
        if a is None or b is None:
            continue
        s1c.append(a); s2c.append(b); sc.append(rr["predicted_prob"]); pids.append(f"{dom}:{rr['puzzle_id']}")
    return (np.array(s1c), np.array(s2c, dtype=float), np.array(sc, dtype=float), np.array(pids))


def boot_area(s1c, s2c, sc, pids):
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(set(pids.tolist())))
    by = {p: np.where(pids == p)[0] for p in uniq}
    ests = []
    for _ in range(N_BOOT):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[p] for p in samp])
        ests.append(curve_area(s1c[idx], s2c[idx], sc[idx])[0])
    ests = np.array(ests)
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))


def main():
    preds = pd.read_parquet(PAPER / "predictions.parquet")
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    p_sub = preds[(preds["config"] == "c") & (preds["feature_subset"] == "canonical_with_sr") &
                  (preds["cv_setup"] == "lopo_within_domain")]
    aug = traces[traces["pass_type"] == "augmented"]
    correct = {(r.model, r.puzzle_id, r.domain): int(r.correct) for r in aug.itertuples()}

    print("Routing value = mean(classifier_acc - random_line) over the escalation sweep, in pp.")
    print("Positive & CI>0  =>  classifier routes better than random (chance) escalation.\n")
    print(f"{'Domain':<16s} {'n':>4s} {'Aflash':>7s} {'Apro':>6s} {'area_pp':>8s} "
          f"{'95% CI (pp)':>18s} {'sig':>4s} {'maxgap_pp@esc':>15s} {'esc→Pro':>8s}")
    print("-" * 92)

    all_s1, all_s2, all_sc, all_pid = [], [], [], []
    for dom in DOMAINS:
        s1c, s2c, sc, pids = load_domain(p_sub, correct, dom)
        if len(s1c) == 0:
            print(f"{dom}: no data"); continue
        area, mg, mr, e2p, af, ap = curve_area(s1c, s2c, sc)
        lo, hi = boot_area(s1c, s2c, sc, pids)
        sig = "*" if (lo > 0 or hi < 0) else ""
        print(f"{dom:<16s} {len(s1c):>4d} {af:>7.3f} {ap:>6.3f} {area*100:>8.2f} "
              f"[{lo*100:>+6.2f}, {hi*100:>+6.2f}] {sig:>4s} {mg*100:>8.2f}@{mr*100:>4.0f}% {e2p*100:>7.0f}%")
        all_s1.append(s1c); all_s2.append(s2c); all_sc.append(sc); all_pid.append(pids)

    s1c = np.concatenate(all_s1); s2c = np.concatenate(all_s2)
    sc = np.concatenate(all_sc); pids = np.concatenate(all_pid)
    area, mg, mr, e2p, af, ap = curve_area(s1c, s2c, sc)
    lo, hi = boot_area(s1c, s2c, sc, pids)
    sig = "*" if (lo > 0 or hi < 0) else ""
    print("-" * 92)
    print(f"{'POOLED':<16s} {len(s1c):>4d} {af:>7.3f} {ap:>6.3f} {area*100:>8.2f} "
          f"[{lo*100:>+6.2f}, {hi*100:>+6.2f}] {sig:>4s} {mg*100:>8.2f}@{mr*100:>4.0f}% {e2p*100:>7.0f}%")
    print("\n* = area CI excludes zero (routing beats random)")


if __name__ == "__main__":
    main()
