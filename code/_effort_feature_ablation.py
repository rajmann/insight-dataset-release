"""Single-feature and leave-one-out (LOO) ablation of the 6 effort-signal features.

Answers reviewer KGMW ("no single-feature ablation exists ... six-feature set arbitrary"):
  SINGLE-FEATURE : train on each of the 6 features ALONE -> AP. Power in isolation.
                   ("which surface signals dominate prediction power?")
  LEAVE-ONE-OUT  : full 6 minus one -> AP, and the drop vs the full set. Marginal
                   contribution GIVEN the other five (exposes redundancy).

Identical methodology to the paper's headline classifier: StandardScaler +
LogisticRegression(C=1.0), leave-one-puzzle-out within domain, AP + lift over base rate.
Reuses EFFORT_SIGNAL and lopo_within_domain from _4domain_effort_signal_classifier.
Insight domains, un_augmented pass (the paper's operating point).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _4domain_effort_signal_classifier import EFFORT_SIGNAL, lopo_within_domain

PAPER = HERE.parent / "data" / "insight_4domain"
DOMAINS = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]

# short labels for display
SHORT = {
    "tokens_thinking_proxy": "tokens_proxy",
    "elapsed": "elapsed",
    "hedge_position_variance": "hedge_pos_var",
    "thinking_char_count": "thinking_chars",
    "hedge_rate": "hedge_rate",
    "hedge_ratio": "hedge_ratio",
}


def load():
    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    if "pass_type" in traces.columns:
        traces = traces[traces["pass_type"] == "un_augmented"]
    feat = pd.read_parquet(PAPER / "features.parquet")
    df = traces[["row_id", "domain", "model", "puzzle_id", "answer", "correct"]
                ].merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["answer"].fillna("").str.len() > 0].reset_index(drop=True)
    return df[df["domain"].isin(DOMAINS)].reset_index(drop=True)


def ap_across(df, features):
    """Per-domain LOPO AP + lift; return dict domain->res and the mean AP/lift."""
    per = {}
    for d in DOMAINS:
        per[d] = lopo_within_domain(df, features, d)
    aps = [per[d]["AP"] for d in DOMAINS if per[d]]
    lifts = [per[d]["lift"] for d in DOMAINS if per[d]]
    return per, float(np.mean(aps)), float(np.mean(lifts))


def main():
    df = load()
    base_rates = {d: df[df.domain == d]["correct"].mean() for d in DOMAINS}
    print(f"Insight, un_augmented. n={len(df)} traces. LOPO within domain. Metric: AP.")
    print("Base rates: " + ", ".join(f"{d} {base_rates[d]:.2f}" for d in DOMAINS) + "\n")

    # FULL 6-feature reference
    full_per, full_ap, full_lift = ap_across(df, EFFORT_SIGNAL)

    # per feature: single-feature (per-domain AP + meanAP) and LOO delta vs FULL
    rows = []
    for f in EFFORT_SIGNAL:
        s_per, s_ap, _ = ap_across(df, [f])                       # alone
        _, l_ap, _ = ap_across(df, [x for x in EFFORT_SIGNAL if x != f])  # drop f
        rows.append((SHORT[f], s_per, s_ap, l_ap - full_ap))
    rows.sort(key=lambda r: -r[2])  # by solo power, strongest first

    hdr = (f"{'feature':16s} | " + " | ".join(f"{d[:6]:>6s}" for d in DOMAINS) +
           f" | {'alone':>6s} | {'LOO d':>6s}")
    print("COMBINED ABLATION (single-feature per-domain AP + solo meanAP + LOO delta)")
    print(f"FULL 6 reference: meanAP {full_ap:.3f}, lift {full_lift:+.3f}\n")
    print(hdr); print("-" * len(hdr))
    for name, s_per, s_ap, loo_d in rows:
        cells = " | ".join(f"{(s_per[d]['AP'] if s_per[d] else float('nan')):6.3f}" for d in DOMAINS)
        print(f"{name:16s} | {cells} | {s_ap:6.3f} | {loo_d:+6.3f}")
    print("-" * len(hdr))
    print("\nalone = AP with that feature as sole predictor (higher = stronger in isolation).")
    print("LOO d = change in mean AP when the feature is removed from the six")
    print("        (nearer zero = more redundant; most negative = most load-bearing).")
    print("Divergence is the story: e.g. hedge features are middling alone yet ~0 to drop.")


if __name__ == "__main__":
    main()
