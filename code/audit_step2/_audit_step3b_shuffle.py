"""Step 3b: trace-position robustness of the deployed per-candidate judge.

Re-judges the 75 gold-standard traces with the SAME judge (gemini-3-flash-preview,
thinking_budget=0) and the SAME candidate list, but with the trace text reordered
(paragraph or sentence units shuffled). If the judge scores candidates the same
way after the conclusion is moved off the end of the trace, its scores are driven
by the language around a candidate, not by the candidate appearing last.

Design notes (why this is confound-free):
  - Candidate LIST order is held at the deployed judge's original input order
    (reconstructed from eureka_judge_*.csv), so this isolates TRACE position from
    candidate-list position (the separate Step 3 question).
  - Trace text is the exact truncated text the judge saw (from sample.json).
  - One permutation per run (K=1). Re-run with a different --seed to add an
    independent permutation; pool offline.
  - Coherence confound: sentence shuffling destroys coherence, paragraph shuffling
    largely preserves it. Run both granularities; if scores hold under paragraph
    but drop under sentence, the drop is coherence loss, not position. The
    chosen-vs-non-chosen shift localises any genuine position effect to the
    conclusion.

Usage:
  python _audit_step3b_shuffle.py --granularity paragraph --seed 0
  python _audit_step3b_shuffle.py --granularity sentence  --seed 0
Env: GEMINI_API_KEY (same as the deployed judge).
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parents[1]            # repo root (rebus/)
sys.path.insert(0, str(ROOT))
from _eureka_llm_judge import PROMPT_TEMPLATE, call_gemini   # reuse exact prompt + call

# Key handling follows the existing scripts (openrouter_phase2.py): load from .env
# rather than requiring an exported shell var. Try the locations those scripts use.
from dotenv import load_dotenv
for _envp in (HERE.parent / ".env", ROOT / ".env"):
    if _envp.exists():
        load_dotenv(_envp)


def split_units(text: str, granularity: str) -> list[str]:
    if granularity == "paragraph":
        return [u for u in re.split(r"\n\s*\n", text) if u.strip()]
    if granularity == "sentence":
        return [u for u in re.split(r"(?<=[.!?])\s+|\n+", text) if u.strip()]
    raise ValueError(granularity)


def shuffle_text(text: str, granularity: str, rng: random.Random):
    units = split_units(text, granularity)
    can = len(units) >= 3                       # need >=3 units for a meaningful reorder
    if not can:
        return text, len(units), False
    order = list(range(len(units)))
    for _ in range(8):                           # avoid the identity permutation
        rng.shuffle(order)
        if order != sorted(order):
            break
    joiner = "\n\n" if granularity == "paragraph" else " "
    return joiner.join(units[i] for i in order), len(units), True


def candidate_order_by_trace() -> dict[str, list[str]]:
    """Reconstruct the deployed judge's per-trace candidate input order from its
    own output CSVs (row order = input order)."""
    jf = pd.read_csv(ROOT / "eureka_judge_full.csv"); jf["pass_type"] = "un_augmented"
    vp = pd.read_csv(ROOT / "eureka_judge_vp.csv")
    out = {}
    for df in (jf, vp):
        for key, g in df.groupby(["model", "domain", "puzzle_id", "pass_type"], sort=False):
            model, domain, pid, ptype = key
            tid = f"{domain}|{model}|{pid}|{ptype}"
            cands, seen = [], set()
            for c in g["candidate"].tolist():
                if isinstance(c, str) and c.strip() and c not in seen:
                    seen.add(c); cands.append(c)
            out[tid] = cands
    return out


def generate(granularity: str, seed: int, workers: int) -> Path:
    sample = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))
    order = candidate_order_by_trace()
    rng = random.Random(seed)

    jobs = []
    for t in sample["traces"]:
        tid = t["trace_id"]
        cands = order.get(tid) or t["candidates"]
        shuffled, n_units, can = shuffle_text(t["trace_text"], granularity, rng)
        jobs.append((tid, cands, shuffled, n_units, can))

    n_shuf = sum(1 for j in jobs if j[4])
    print(f"{len(jobs)} traces | {n_shuf} shuffleable at {granularity} granularity "
          f"({len(jobs)-n_shuf} have <3 units, sent unchanged)")

    def one(job):
        tid, cands, shuffled, n_units, can = job
        prompt = PROMPT_TEMPLATE.format(
            candidate_list="\n".join(f"- {c}" for c in cands), trace=shuffled)
        try:
            parsed, raw = call_gemini(prompt)
        except Exception as e:
            return tid, None, f"call_failed: {e}", n_units, can
        if not parsed or "candidates" not in parsed:
            return tid, None, f"parse_failed: {str(raw)[:120]!r}", n_units, can
        scores = {}
        for o in parsed["candidates"]:
            c = o.get("candidate"); conf = o.get("confidence")
            if isinstance(c, str):
                scores[c] = conf
        return tid, scores, None, n_units, can

    results, errors, done = {}, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            tid, scores, err, n_units, can = fut.result()
            done += 1
            if err:
                errors += 1
                print(f"  [{done}/{len(jobs)}] {tid} {err[:80]}")
            else:
                results[tid] = {"scores": scores, "n_units": n_units, "can_shuffle": can}
            if done % 20 == 0:
                print(f"  [{done}/{len(jobs)}] done, {errors} errors")

    out_path = DATA / f"shuffle_{granularity}_seed{seed}.json"
    out_path.write_text(json.dumps(
        {"meta": {"granularity": granularity, "seed": seed, "n_traces": len(results)},
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  ({len(results)} traces, {errors} errors)")
    return out_path


def compare(shuffle_path: Path):
    shuf = json.loads(shuffle_path.read_text(encoding="utf-8"))["results"]
    key = json.loads((DATA / "gold_key.json").read_text(encoding="utf-8"))
    ann = json.loads((DATA / "annotations.json").read_text(encoding="utf-8")) \
        if (DATA / "annotations.json").exists() else {}

    rows = []
    for tid, r in shuf.items():
        if not r.get("can_shuffle"):
            continue                                  # only manipulated traces
        ck = key.get(tid, {}).get("candidates", {})
        human = (ann.get(tid, {}).get("scores") or {})
        for c, sconf in r["scores"].items():
            base = ck.get(c, {}).get("judge_confidence")
            if base is None or sconf is None:
                continue
            rows.append(dict(tid=tid, cand=c, base=float(base), shuf=float(sconf),
                             chosen=ck.get(c, {}).get("is_chosen", 0),
                             human=human.get(c)))
    if len(rows) < 5:
        print("not enough comparable pairs"); return

    base = np.array([r["base"] for r in rows]); shufv = np.array([r["shuf"] for r in rows])
    rho = spearmanr(base, shufv).statistic
    shift = shufv - base
    print(f"\n=== Trace-position robustness ({shuffle_path.name}) ===")
    print(f"comparable pairs (shuffleable traces only): {len(rows)}")
    print(f"original-judge vs shuffled-judge: Spearman rho = {rho:.3f}")
    print(f"mean signed shift (shuffled - original): {shift.mean():+.1f} (median {np.median(shift):+.0f})")

    ch = np.array([r["shuf"] - r["base"] for r in rows if r["chosen"]])
    nc = np.array([r["shuf"] - r["base"] for r in rows if not r["chosen"]])
    print(f"  shift on CHOSEN (n={len(ch)}): {ch.mean():+.1f}   shift on NON-chosen (n={len(nc)}): {nc.mean():+.1f}")
    print("  (chosen-specific drop = genuine position effect on the conclusion; "
          "uniform drop = coherence degradation)")

    hr = [r for r in rows if r.get("human") is not None]
    if hr:
        b = np.mean([abs(r["base"] - r["human"]) for r in hr])
        s = np.mean([abs(r["shuf"] - r["human"]) for r in hr])
        print(f"  mean |judge - human|: original {b:.1f} -> shuffled {s:.1f} "
              f"({'closer to human' if s < b else 'further from human'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", choices=["paragraph", "sentence"], default="paragraph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare-only", default=None,
                    help="Path to an existing shuffle_*.json to (re)compare without re-judging")
    args = ap.parse_args()

    if args.compare_only:
        compare(Path(args.compare_only)); return
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")):
        raise SystemExit("Set GEMINI_API_KEY (same as the deployed judge) before running.")
    out = generate(args.granularity, args.seed, args.workers)
    compare(out)


if __name__ == "__main__":
    main()
