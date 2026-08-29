"""Appendix figure: continuous gains / losses / net escalation curves for reconsideration.

Uses ALL same-snapshot traces (scored-consistent), ordered by classifier confidence
(lowest first). x = fraction re-prompted; y = whole-set accuracy change (pp).
Panel (a): pooled gains / losses / net with a puzzle-clustered bootstrap band on net.
Panel (b): net by domain (pooled / Cryptic / Rebus).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _reprompt_control_ci import build

OUT = HERE.parent / "diagrams" / "reprompt_escalation.png"
GRID = np.linspace(0.0, 1.0, 101)[1:]           # 0.01 .. 1.00
RNG = np.random.default_rng(0)
NB = 600
# Okabe-Ito colourblind-safe
C = {"gains": "#009E73", "losses": "#D55E00", "net": "#000000",
     "Cryptic": "#0072B2", "Rebus": "#D55E00", "pooled": "#000000"}


def curves(sub):
    """cumulative gains, losses, net (pp of whole set) on GRID, ordered lowest-conf first."""
    s = sub.sort_values("pp")
    g = s["gain"].to_numpy()
    n = len(g)
    rec = np.cumsum(g == 1); harm = np.cumsum(g == -1)
    idx = np.clip((GRID * n).astype(int) - 1, 0, n - 1)
    return rec[idx] / n * 100, harm[idx] / n * 100


def net_band(sub):
    puz = sub["puzzle_id"].unique()
    loc = {p: sub.index[sub.puzzle_id == p].to_numpy() for p in puz}
    mat = np.empty((NB, len(GRID)))
    for b in range(NB):
        pick = RNG.choice(puz, len(puz), replace=True)
        rr = sub.loc[np.concatenate([loc[p] for p in pick])]
        gpp, lpp = curves(rr)
        mat[b] = gpp - lpp
    return np.percentile(mat, [2.5, 97.5], axis=0)


def main():
    df = build()
    d = df[~df["sub"]]
    gpp, lpp = curves(d)
    net = gpp - lpp
    lo, hi = net_band(d)
    peak = GRID[np.argmax(net)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    x = GRID * 100
    ax1.axhline(0, color="#888", lw=0.7)
    ax1.plot(x, gpp, color=C["gains"], lw=2, label="gains (recoveries)")
    ax1.plot(x, -lpp, color=C["losses"], lw=2, label="losses (harm)")
    ax1.plot(x, net, color=C["net"], lw=2.2, label="net")
    ax1.fill_between(x, lo, hi, color="#000000", alpha=0.12, lw=0)
    ax1.axvline(peak * 100, color="#888", ls=":", lw=1)
    ax1.set_title("(a) Pooled: gains front-loaded, losses back-loaded", fontsize=9)
    ax1.set_xlabel("Re-prompted (%, lowest-confidence first)", fontsize=9)
    ax1.set_ylabel("Whole-set accuracy change (pp)", fontsize=9)
    ax1.legend(fontsize=8, frameon=False, loc="upper left")

    ax2.axhline(0, color="#888", lw=0.7)
    for dom, col in [("pooled", C["pooled"]), ("Cryptic", C["Cryptic"]), ("Rebus", C["Rebus"])]:
        sub = d if dom == "pooled" else d[d.domain == dom]
        g2, l2 = curves(sub)
        ax2.plot(x, g2 - l2, color=col, lw=2, label=dom,
                 ls="-" if dom == "pooled" else "--")
    ax2.set_title("(b) Net by domain", fontsize=9)
    ax2.set_xlabel("Re-prompted (%, lowest-confidence first)", fontsize=9)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}  | pooled net peak at {peak*100:.0f}% = {net.max():+.2f}pp")


if __name__ == "__main__":
    main()
