"""Math vs insight, laid out cleanly on the RIGHT corpus (hard_math_93, post-cutoff).

The confusion we are untangling: we have been mixing two orthogonal questions per feature.
  PREVALENCE   - how much does the marker/feature APPEAR in the domain? (median value)
  DISCRIMINATION - does the feature separate CORRECT from WRONG within the domain?
                   (Spearman rho with correctness; and wrong/correct median ratio for effort)

A feature can be present but non-discriminative, or rare, or both. Table 2 in the paper only
showed DISCRIMINATION (rho), on the WRONG corpus (MATH-500). Here: both questions, hard_math_93.

Feature groups:
  EFFORT (magnitude of thinking)        - tokens, elapsed, chars.   Present in both domains.
  HEDGE (uncertainty language)          - hedge rate/ratio/pos-var. Canonical classifier features.
  OTHER MARKERS (resolution language)   - rejection, eureka, assertion densities.
  [chosen_conf / local confidence]      - insight-only by construction; noted, not in this table.

Insight: features.parquet (paper's canonical extraction) + all_traces (correct, thinking).
Math: hard_math_93/results_un_aug, scored with _hard_math_score_v2, features via extract_features_B.
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
from train_confidence_classifier import (extract_features_B, HEDGE_PATTERNS,
                                          CONFIDENCE_PATTERNS, EXPLICIT_REJECTION_PATTERNS,
                                          count_patterns)
from _hard_math_score_v2 import score_row

PAPER = HERE.parent / "data" / "insight_4domain"
INSIGHT = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]
MROOT = HERE.parent / "data" / "hard_math_93" / "results_un_aug"
SAFE = {"Gemini 2.5 Pro": "Gemini_2_5_Pro", "Gemini 3 Flash": "Gemini_3_Flash",
        "Gemini 3 Pro": "Gemini_3_Pro", "Qwen3-VL-235B Thinking": "Qwen3_VL_235B_Thinking",
        "GPT-5": "GPT_5", "Claude Opus 4.6": "Claude_Opus_4_6", "Claude Sonnet 4.6": "Claude_Sonnet_4_6"}
EUREKA = [r"\bthat's it\b", r'\beureka\b', r'\byes!\b', r'\bperfect\b', r'\bgot it\b',
          r'\bjumps( out)?\b', r'\bright!', r'\bright off\b', r'\bspot on\b']
ASSERT = [p for p in CONFIDENCE_PATTERNS if p not in EUREKA]
MARKER_FAMILIES = {"hedge": HEDGE_PATTERNS, "rejection": EXPLICIT_REJECTION_PATTERNS,
                   "eureka": EUREKA, "assertion": ASSERT}


def dens(text, pats):
    return count_patterns(text, pats) / max(len(text.split()), 1) * 100


def proxy(raw, thinking):
    r = int(raw or 0)
    return float(r) if r > 0 else len(thinking or "") / 4.0


# ---------- load insight (canonical features + text) ----------
def load_insight():
    tr = pd.read_parquet(PAPER / "all_traces.parquet")
    tr = tr[(tr["pass_type"] == "un_augmented") & (tr["answer"].fillna("").str.len() > 0)]
    feat = pd.read_parquet(PAPER / "features.parquet")
    base = tr[["row_id", "domain", "model", "puzzle_id", "correct", "thinking",
               "tokens_thinking", "elapsed_seconds"]].rename(columns={"tokens_thinking": "raw_tok"})
    feat = feat.drop(columns=[c for c in ("tokens_thinking", "elapsed") if c in feat.columns])
    df = base.merge(feat, on="row_id", how="left")
    df["correct"] = df["correct"].astype(int)
    df = df[df["domain"].isin(INSIGHT)].reset_index(drop=True)
    df["tok_proxy"] = [proxy(r.raw_tok, r.thinking) for r in df.itertuples()]
    df["chars"] = df["thinking"].str.len()
    df["elapsed_s"] = df["elapsed_seconds"].astype(float)
    for fam, pats in MARKER_FAMILIES.items():
        df[f"dens_{fam}"] = [dens(t, pats) for t in df["thinking"]]
    # chosen_conf: local confidence over the CHOSEN answer, extracted via candidate
    # detection. Motivated by the insight aha-moment; no math analogue by construction.
    ff = pd.read_parquet(HERE / "filtration_candidates" / "filtration_features.parquet")
    ff = ff[(ff["domain"].isin(INSIGHT)) & (ff["pass_type"] == "un_augmented")]
    df = df.merge(ff[["row_id", "chosen_conf_mean"]], on="row_id", how="left")
    return df


# ---------- load math (score v2 + recompute features) ----------
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
            row = {"correct": c, "puzzle_id": r["puzzle_id"], "thinking": th,
                   "tok_proxy": proxy(r.get("tokens_thinking"), th),
                   "chars": len(th), "elapsed_s": float(r.get("elapsed") or 0),
                   "hedge_rate": count_patterns(th, HEDGE_PATTERNS) / max(len(th.split()), 1),
                   "hedge_ratio": fb.get("hedge_ratio", np.nan),
                   "hedge_position_variance": fb.get("hedge_position_variance", np.nan)}
            for fam, pats in MARKER_FAMILIES.items():
                row[f"dens_{fam}"] = dens(th, pats)
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(v):
    """Adaptive precision: big magnitudes as integers, small (rates/variances) with 4 dp
    so a real 0.008 is not rounded to a meaningless 0.01/0.00."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "   n/a"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def rho(sub, col):
    d = sub.dropna(subset=[col])
    if len(d) < 10 or d["correct"].std() == 0 or d[col].std() == 0:
        return np.nan
    return spearmanr(d[col], d["correct"]).statistic


def insight_meanrho(ins, col):
    vals = [rho(ins[ins.domain == d], col) for d in INSIGHT]
    return np.nanmean([v for v in vals if not np.isnan(v)])


def med(sub, col):
    return float(np.nanmedian(sub[col].astype(float)))


def wc_ratio(sub, col):
    w = sub[sub.correct == 0][col].astype(float).dropna()
    c = sub[sub.correct == 1][col].astype(float).dropna()
    if len(w) < 3 or len(c) < 3 or np.nanmedian(c) == 0:
        return np.nan
    return np.nanmedian(w) / np.nanmedian(c)


def rho_p(sub, col):
    """Spearman rho and p-value within a (already-subset) frame."""
    d = sub.dropna(subset=[col])
    if len(d) < 10 or d["correct"].std() == 0 or d[col].std() == 0:
        return np.nan, np.nan
    r = spearmanr(d[col], d["correct"])
    return r.statistic, r.pvalue


def table2_row(ins, math, col):
    """Insight mean-rho (star if all 4 domains p<0.05) and math rho (star if p<0.05)."""
    per = [rho_p(ins[ins.domain == d], col) for d in INSIGHT]
    ir = np.nanmean([r for r, _ in per if not np.isnan(r)])
    istar = "*" if all((not np.isnan(p)) and p < 0.05 for _, p in per) else " "
    mr, mp = rho_p(math, col)
    mstar = "*" if (not np.isnan(mp)) and mp < 0.05 else " "
    return ir, istar, mr, mstar


def med_wc(sub, col):
    """(median-when-wrong, median-when-correct) for a feature within a domain."""
    d = sub.dropna(subset=[col])
    w = d[d.correct == 0][col].astype(float)
    c = d[d.correct == 1][col].astype(float)
    if len(w) < 3 or len(c) < 3:
        return np.nan, np.nan
    return float(np.median(w)), float(np.median(c))


def main():
    ins, math = load_insight(), load_math()
    print(f"Insight n={len(ins)} (4 domains, un_augmented). "
          f"Math n={len(math)} (hard_math_93, scored v2, accuracy {math['correct'].mean():.2f}; NOT MATH-500).")
    print("rho = Spearman(feature, correct); rho<0 means a higher feature value goes with being WRONG.\n")

    # each entry: (column, human label, group)
    FEATS = [("tok_proxy", "token count", "EFFORT (magnitude of thinking)"),
             ("elapsed_s", "elapsed seconds", "EFFORT (magnitude of thinking)"),
             ("chars", "character count", "EFFORT (magnitude of thinking)"),
             ("hedge_rate", "hedges per word", "HEDGE (uncertainty language)"),
             ("hedge_ratio", "hedge fraction of tokens", "HEDGE (uncertainty language)"),
             ("hedge_position_variance", "hedge spread in trace", "HEDGE (uncertainty language)"),
             ("chosen_conf_mean", "confidence in chosen answer", "LOCAL CONFIDENCE (aha signal)")]

    # ---- PART 1: prevalence + discrimination
    print("=" * 100)
    print("PART 1  Per feature: PREVALENCE (median value) and DISCRIMINATION (Spearman rho with correctness)")
    print("        Table 2 replacement, now on hard_math_93 (post-cutoff), not MATH-500.")
    print("=" * 100)
    hdr = (f"{'feature':26s} | {'measures':26s} | {'Insight median':>14s} {'Insight ρ':>10s} | "
           f"{'Math median':>12s} {'Math ρ':>8s}")
    print(hdr); print("-" * len(hdr))
    last_grp = None
    for col, meas, grp in FEATS:
        if grp != last_grp:
            print(f"{grp}"); last_grp = grp
        im, ir = med(ins, col), insight_meanrho(ins, col)
        if col == "chosen_conf_mean":
            mm, mr = np.nan, np.nan   # insight-motivated; no math analogue by construction
        else:
            mm, mr = med(math, col), rho(math, col)
        print(f"  {col:24s} | {meas:26s} | {fmt(im):>14s} {fmt(ir):>10s} | {fmt(mm):>12s} {fmt(mr):>8s}")
    print("\n  chosen_conf has no math value by construction: it is the model's expressed confidence in the")
    print("  CHOSEN candidate answer, motivated by the insight aha-moment. Math answers are not candidate")
    print("  selections, so the signal is insight-specific by design - its absence in math is the point.")

    # ---- TABLE 2 RECOMPUTE: paper format (Insight mean | Math), hard_math_93, with significance
    print("\n" + "=" * 100)
    print("TABLE 2 RECOMPUTE (paper format): Spearman rho(feature, correct). * = p<0.05.")
    print("  Insight = mean across 4 domains (* if all 4 significant); Math = hard_math_93 (post-cutoff).")
    print("=" * 100)
    print(f"{'feature':26s} | {'Insight (mean)':>16s} | {'Math (hard_math_93)':>20s}")
    print("-" * 70)
    for col, _, _ in FEATS:
        if col == "chosen_conf_mean":
            continue
        ir, istar, mr, mstar = table2_row(ins, math, col)
        print(f"{col:26s} | {ir:>13.2f}{istar}   | {mr:>17.2f}{mstar}")

    # ---- PART 2: marker families prevalence vs discrimination
    print("\n" + "=" * 100)
    print("PART 2  Marker families: does the marker APPEAR (density) vs does it PREDICT (rho)?")
    print("        density = markers per 100 words (median).")
    print("=" * 100)
    hdr2 = (f"{'marker family':16s} | {'Insight density':>15s} {'Math density':>13s} {'Insight/Math':>13s} | "
            f"{'Insight ρ':>10s} {'Math ρ':>8s}")
    print(hdr2); print("-" * len(hdr2))
    for fam in MARKER_FAMILIES:
        col = f"dens_{fam}"
        idens, mdens = med(ins, col), med(math, col)
        ratio = f"{idens/mdens:.1f}x" if mdens > 0 else "  --"
        ir, mr = insight_meanrho(ins, col), rho(math, col)
        print(f"{fam:16s} | {fmt(idens):>15s} {fmt(mdens):>13s} {ratio:>13s} | {fmt(ir):>10s} {fmt(mr):>8s}")

    # ---- PART 3: wrong vs correct within each domain (the magnitude lens)
    print("\n" + "=" * 100)
    print("PART 3  WRONG vs CORRECT traces within each domain (median), and the wrong/correct ratio.")
    print("        Shows the magnitude gap directly (>1 = wrong-answer traces score higher on the feature).")
    print("=" * 100)
    hdr3 = (f"{'feature':26s} | {'Insight wrong':>13s} {'Insight correct':>15s} {'ratio':>7s} | "
            f"{'Math wrong':>11s} {'Math correct':>13s} {'ratio':>7s}")
    print(hdr3); print("-" * len(hdr3))
    for col, meas, grp in FEATS:
        iw, ic = med_wc(ins, col)
        iratio = f"{iw/ic:.2f}x" if ic else "  n/a"
        if col == "chosen_conf_mean":
            mw = mc = np.nan; mratio = "  n/a"
        else:
            mw, mc = med_wc(math, col)
            mratio = f"{mw/mc:.2f}x" if mc else "  n/a"
        print(f"  {col:24s} | {fmt(iw):>13s} {fmt(ic):>15s} {iratio:>7s} | "
              f"{fmt(mw):>11s} {fmt(mc):>13s} {mratio:>7s}")


if __name__ == "__main__":
    main()
