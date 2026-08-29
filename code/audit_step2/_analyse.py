"""Step 2 audit analysis: compare my blinded gold-standard scores against the
deployed judge's stored judge_confidence.

Reads:
  data/sample.json        - trace -> candidate list, domain, puzzle_id
  data/gold_key.json       - hidden key: per-candidate judge_confidence + ngrams
  data/annotations.json    - my scores + ngrams from the labelling tool

Computes:
  - Spearman rho (pooled + per-domain) over (trace, candidate) pairs
  - Pearson r (pooled)
  - Bootstrap 95% CI on pooled Spearman, resampling CLUSTERS = (domain, puzzle_id)
    to respect multiple models sharing a puzzle
  - Quartile-bucket Cohen's quadratic-weighted kappa
  - Per-judge-bucket disagreement (where judge says 80+, my distribution)
  - Largest-disagreement examples with both sides' n-grams (for the writeup)

Usage:
  python _analyse.py [--n-boot 2000] [--seed 0]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

VERDICTS = [(0.7, "rho > 0.7: judge tracks human intuition; methodology defensible without Step 3."),
            (0.5, "0.5 < rho < 0.7: noisy but monotonic; caveat in rebuttal, address as follow-up."),
            (0.0, "rho < 0.5: real misalignment; rerun judge with improved methodology.")]


def load_pairs():
    sample = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    key = json.loads((DATA / "gold_key.json").read_text(encoding="utf-8"))
    annp = DATA / "annotations.json"
    ann = json.loads(annp.read_text(encoding="utf-8")) if annp.exists() else {}

    rows = []
    flagged = []        # candidates marked "not a real candidate" (extraction noise)
    for t in sample["traces"]:
        tid = t["trace_id"]
        a = ann.get(tid, {})
        my_scores = a.get("scores", {}) or {}
        my_ng = a.get("ngrams", {}) or {}
        my_flags = a.get("flags", {}) or {}
        ck = key.get(tid, {}).get("candidates", {})
        for cand in t["candidates"]:
            is_flag = bool(my_flags.get(cand))
            jc = ck.get(cand, {}).get("judge_confidence")
            if is_flag:
                flagged.append({"trace_id": tid, "domain": t["domain"],
                                "candidate": cand, "judge": jc})
            mine = my_scores.get(cand)
            if mine is None or jc is None:
                continue
            rows.append({
                "trace_id": tid, "domain": t["domain"], "puzzle_id": t["puzzle_id"],
                "model": t["model"], "candidate": cand,
                "mine": float(mine), "judge": float(jc), "flagged": is_flag,
                "is_chosen": ck.get(cand, {}).get("is_chosen", 0),
                "my_ngrams": my_ng.get(cand, []),
                "judge_ngrams": ck.get(cand, {}).get("judge_ngrams", "[]"),
            })
    return rows, sample, ann, flagged


def quad_weighted_kappa(a, b, k=4):
    """Cohen's quadratic-weighted kappa on quartile buckets 0..k-1."""
    a = np.asarray(a); b = np.asarray(b)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[x, y] += 1
    w = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            w[i, j] = (i - j) ** 2 / (k - 1) ** 2
    ah = O.sum(axis=1); bh = O.sum(axis=0); n = O.sum()
    if n == 0:
        return float("nan")
    E = np.outer(ah, bh) / n
    num = (w * O).sum(); den = (w * E).sum()
    return 1.0 - num / den if den else float("nan")


def bucket(x):
    return min(3, int(x) // 25)   # 0-24->0, 25-49->1, 50-74->2, 75-100->3


def bootstrap_cluster_spearman(rows, n_boot, seed):
    rng = np.random.default_rng(seed)
    clusters = {}
    for r in rows:
        clusters.setdefault((r["domain"], r["puzzle_id"]), []).append(r)
    keys = list(clusters.keys())
    est = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        mine, judge = [], []
        for idx in pick:
            for r in clusters[keys[idx]]:
                mine.append(r["mine"]); judge.append(r["judge"])
        if len(set(mine)) > 1 and len(set(judge)) > 1:
            est.append(spearmanr(mine, judge).statistic)
    est = np.array(est)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), len(est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, sample, ann, flagged = load_pairs()
    total_pairs = sum(len(t["candidates"]) for t in sample["traces"])

    def _handled(tid, cands):
        r = ann.get(tid, {}); sc = r.get("scores") or {}; fl = r.get("flags") or {}
        return all((sc.get(c) is not None) or fl.get(c) for c in cands)
    n_done = sum(1 for t in sample["traces"]
                 if (ann.get(t["trace_id"], {}).get("scores") or ann.get(t["trace_id"], {}).get("flags"))
                 and _handled(t["trace_id"], t["candidates"]))
    print(f"=== Step 2 audit analysis ===")
    print(f"Traces fully scored: {n_done}/{len(sample['traces'])}")
    print(f"Comparable (trace,candidate) pairs: {len(rows)} / {total_pairs} possible\n")
    if len(rows) < 10:
        print("Not enough scored pairs yet. Score more traces in the tool, then re-run.")
        return

    mine = np.array([r["mine"] for r in rows])
    judge = np.array([r["judge"] for r in rows])

    # pooled
    rho = spearmanr(mine, judge).statistic
    pr = pearsonr(mine, judge).statistic
    lo, hi, nb = bootstrap_cluster_spearman(rows, args.n_boot, args.seed)
    kap = quad_weighted_kappa([bucket(x) for x in mine], [bucket(x) for x in judge])

    print(f"POOLED  Spearman rho = {rho:.3f}   (95% CI [{lo:.3f}, {hi:.3f}], {nb} cluster-bootstrap reps)")
    print(f"        Pearson r    = {pr:.3f}")
    print(f"        Quartile quadratic-weighted Cohen's kappa = {kap:.3f}")
    # sensitivity: exclude candidates flagged as not-real (extraction noise)
    genuine = [r for r in rows if not r.get("flagged")]
    rho_gen = None
    if any(r.get("flagged") for r in rows) and len(genuine) > 2:
        gm = np.array([r["mine"] for r in genuine]); gj = np.array([r["judge"] for r in genuine])
        rho_gen = spearmanr(gm, gj).statistic
        print(f"        Spearman rho (genuine candidates only, {len(genuine)} pairs) = {rho_gen:.3f}")
    print()

    # --- the crux: chosen candidate (what chosen_conf is computed from) vs the rest ---
    def _rho(rs):
        if len(rs) < 5:
            return None
        return spearmanr([r["mine"] for r in rs], [r["judge"] for r in rs]).statistic
    chosen = [r for r in rows if r.get("is_chosen")]
    nonchosen = [r for r in rows if not r.get("is_chosen")]
    rho_chosen, rho_nonchosen = _rho(chosen), _rho(nonchosen)
    print("CHOSEN vs NON-CHOSEN (chosen is the candidate that feeds chosen_conf downstream):")
    print(f"  chosen candidate only : rho={rho_chosen if rho_chosen is None else round(rho_chosen,3)}  (n={len(chosen)})")
    print(f"  non-chosen only       : rho={rho_nonchosen if rho_nonchosen is None else round(rho_nonchosen,3)}  (n={len(nonchosen)})")
    d = np.array([r["judge"] - r["mine"] for r in rows])
    print(f"  directional bias (judge - mine): mean={d.mean():+.1f} median={np.median(d):+.0f} "
          f"| judge>mine {100*(d>0).mean():.0f}% (by>=30: {100*(d>=30).mean():.0f}%) "
          f"| mine>judge {100*(d<0).mean():.0f}%")
    if chosen and nonchosen:
        dc = np.mean([r["judge"]-r["mine"] for r in chosen])
        dn = np.mean([r["judge"]-r["mine"] for r in nonchosen])
        print(f"  bias on chosen={dc:+.1f}  vs  non-chosen={dn:+.1f}  "
              f"(larger positive bias on non-answer 'building-block' words = judge scores local positivity, not confidence-as-answer)\n")

    print("PER DOMAIN:")
    per_domain = {}
    for dom in sorted({r["domain"] for r in rows}):
        sub = [r for r in rows if r["domain"] == dom]
        m = np.array([r["mine"] for r in sub]); j = np.array([r["judge"] for r in sub])
        d_rho = spearmanr(m, j).statistic if len(set(m)) > 1 and len(set(j)) > 1 else float("nan")
        per_domain[dom] = {"n_pairs": len(sub), "spearman": d_rho}
        print(f"  {dom:8s}  n={len(sub):4d}  rho={d_rho:.3f}")

    if flagged:
        print("\nEXTRACTION QUALITY - candidates flagged 'not a real candidate':")
        by_dom = {}
        for f in flagged:
            by_dom.setdefault(f["domain"], []).append(f)
        annotated_cands = sum(len(t["candidates"]) for t in sample["traces"]
                              if ann.get(t["trace_id"], {}).get("scores")
                              or ann.get(t["trace_id"], {}).get("flags"))
        print(f"  total flagged: {len(flagged)} / {annotated_cands} candidates in annotated traces "
              f"({100*len(flagged)/max(annotated_cands,1):.0f}%)")
        for dom in sorted(by_dom):
            print(f"    {dom:8s}: {len(by_dom[dom])}")
        jvals = [f["judge"] for f in flagged if f["judge"] is not None]
        if jvals:
            jv = np.array(jvals)
            print(f"  judge scores GIVEN to these non-candidates: n={len(jv)} (rest were judge-null) "
                  f"mean={jv.mean():.1f} median={np.median(jv):.0f} max={jv.max():.0f} "
                  f"| >=60: {int((jv>=60).sum())}  >=80: {int((jv>=80).sum())}")
            print("  (high judge scores on flagged non-candidates = extractor noise propagating "
                  "into spurious confidence; this is a finding about _per_candidate_confidence.py, not the judge.)")

    print("\nDISAGREEMENT where JUDGE >= 80 (eureka claims):")
    hi_j = [r for r in rows if r["judge"] >= 80]
    if hi_j:
        mm = np.array([r["mine"] for r in hi_j])
        print(f"  n={len(hi_j)}  my score: mean={mm.mean():.1f}  median={np.median(mm):.0f}  "
              f"min={mm.min():.0f}  pct<60={100*(mm<60).mean():.0f}%")
    print("DISAGREEMENT where JUDGE <= 20 (rejection claims):")
    lo_j = [r for r in rows if r["judge"] <= 20]
    if lo_j:
        mm = np.array([r["mine"] for r in lo_j])
        print(f"  n={len(lo_j)}  my score: mean={mm.mean():.1f}  median={np.median(mm):.0f}  "
              f"max={mm.max():.0f}  pct>40={100*(mm>40).mean():.0f}%")

    print("\nTOP 8 DISAGREEMENTS (|mine - judge|):")
    for r in sorted(rows, key=lambda r: -abs(r["mine"] - r["judge"]))[:8]:
        try:
            jng = json.loads(r["judge_ngrams"]) if isinstance(r["judge_ngrams"], str) else r["judge_ngrams"]
        except Exception:
            jng = []
        print(f"  [{r['domain']}/{r['model']}/{r['puzzle_id']}] cand={r['candidate'][:40]!r}"
              f"{'  (CHOSEN)' if r['is_chosen'] else ''}")
        print(f"     mine={r['mine']:.0f}  judge={r['judge']:.0f}  |diff|={abs(r['mine']-r['judge']):.0f}")
        print(f"     my ngrams:    {r['my_ngrams']}")
        print(f"     judge ngrams: {jng}")

    verdict = next(v for thr, v in VERDICTS if rho > thr) if rho > 0 else VERDICTS[-1][1]
    print(f"\nVERDICT: {verdict}")
    print("(Headline claim is about DISCRIMINATION, not calibration: monotonic ordering "
          "matters more than absolute offset.)")

    out = {
        "n_traces_done": n_done, "n_traces_total": len(sample["traces"]),
        "n_pairs": len(rows),
        "pooled": {"spearman": rho, "spearman_ci95": [lo, hi], "pearson": pr,
                   "quad_weighted_kappa": kap, "spearman_genuine_only": rho_gen},
        "per_domain": per_domain,
        "chosen_vs_nonchosen": {"spearman_chosen": rho_chosen, "n_chosen": len(chosen),
                                "spearman_nonchosen": rho_nonchosen, "n_nonchosen": len(nonchosen),
                                "mean_signed_bias_judge_minus_mine": float(d.mean())},
        "n_flagged_non_candidates": len(flagged),
    }
    (DATA / "analysis_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {DATA/'analysis_results.json'}")


if __name__ == "__main__":
    main()
