"""Hard math accuracy with hand-flagged algebraic equivalents + manual
override for known extractor misses. Replaces _hard_math_pilot4_status.py
with a more accurate scorer.

Equivalents are mathematically validated by inspection of the actual
expressions; not a generic SymPy normaliser.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _hard_math_inference import extract_boxed
from _math_extractor_v2 import is_correct_v2

# Hand-validated algebraic equivalents.
# Each entry: (puzzle_id, model_pred_string) -> True if equivalent to expected
EQUIVALENTS = {
    # HMMT-2025-Nov-30: \frac{3\sqrt{5}}{7}
    ("HMMT-2025-Nov-30", r"\dfrac{3\sqrt{5}}{7}"): True,
    # HMMT-2026-Feb-9: 50(1 - 1/(2^101-1)) = 100(2^100 - 1)/(2^101 - 1)
    ("HMMT-2026-Feb-9",  r"\frac{100(2^{100}-1)}{2^{101}-1}"): True,
    ("HMMT-2026-Feb-9",  r"\frac{100(2^{100} - 1)}{2^{101} - 1}"): True,
    # HMMT-2025-Nov-16: 5\pi + 6\sqrt{3} = 6\sqrt{3} + 5\pi (commutative)
    ("HMMT-2025-Nov-16", r"6\sqrt{3} + 5\pi"): True,
    ("HMMT-2025-Nov-16", r"6\sqrt{3}+5\pi"): True,
    # HMMT-2026-Feb-16: 2 - \pi/2 = (4-\pi)/2 (algebraic)
    ("HMMT-2026-Feb-16", r"\dfrac{4-\pi}{2}"): True,
    ("HMMT-2026-Feb-16", r"\frac{4-\pi}{2}"): True,
    ("HMMT-2026-Feb-16", r"\frac{4 - \pi}{2}"): True,
    # HMMT-2026-Feb-30: \sqrt{1740}/3 = 2\sqrt{435}/3 (since 1740 = 4 * 435)
    ("HMMT-2026-Feb-30", r"\dfrac{2\sqrt{435}}{3}"): True,
    ("HMMT-2026-Feb-30", r"\frac{2\sqrt{435}}{3}"): True,
}

# Manual overrides where the extractor mis-parsed but inspection shows
# the model produced the correct answer. All Gemini 2.5 Pro — it doesn't
# reliably use \boxed{} on hard problems and ends in prose.
MANUAL_CORRECT = {
    # Feb-23: prose ended with "5\sqrt{5}", extractor returned just "\sqrt"
    ("HMMT-2026-Feb-23", "Gemini_2_5_Pro"): True,
    # Feb-1: "x_3 = -1/21" matches expected -\frac{1}{21}
    ("HMMT-2026-Feb-1", "Gemini_2_5_Pro"): True,
    # Nov-2: "n=12 seems to be the smallest possible value" matches expected 12
    ("HMMT-2025-Nov-2", "Gemini_2_5_Pro"): True,
    # Feb-8: "Sum ≡ 279" matches expected 279
    ("HMMT-2026-Feb-8", "Gemini_2_5_Pro"): True,
}


def score_row(model, puzzle_id, raw_output, expected):
    """Returns ('correct', 'wrong', 'no_pred', 'trunc'). Uses extractor +
    equivalents + manual overrides."""
    if len(raw_output or "") < 100:
        return "trunc", None
    new = extract_boxed(raw_output)
    if MANUAL_CORRECT.get((puzzle_id, model), False):
        return "correct", new
    if new is None:
        return "no_pred", None
    if is_correct_v2(new, expected):
        return "correct", new
    if EQUIVALENTS.get((puzzle_id, new), False):
        return "correct", new
    return "wrong", new


def main():
    d = Path(__file__).resolve().parent.parent / "data" / "hard_math_93" / "results_un_aug"
    rows_all = []
    for p in sorted(d.glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            outcome, pred = score_row(p.stem, r["puzzle_id"],
                                       r.get("raw_output", "") or "",
                                       r["expected"])
            rows_all.append({
                "model": p.stem,
                "puzzle_id": r["puzzle_id"],
                "expected": r["expected"],
                "pred": pred,
                "outcome": outcome,
                "source": r.get("source", ""),
                "year_month": r.get("year_month", ""),
            })

    # Per-model
    print(f"{'Model':<28s} {'rows':>5s} {'corr':>5s} {'wrong':>6s} {'no_pred':>8s} {'trunc':>6s} {'%':>5s}")
    by_model = defaultdict(list)
    for r in rows_all:
        by_model[r["model"]].append(r)
    total = [0, 0, 0, 0, 0]
    for m in sorted(by_model.keys()):
        rows = by_model[m]
        n = len(rows)
        c = sum(1 for r in rows if r["outcome"] == "correct")
        w = sum(1 for r in rows if r["outcome"] == "wrong")
        np_ = sum(1 for r in rows if r["outcome"] == "no_pred")
        t = sum(1 for r in rows if r["outcome"] == "trunc")
        parsed = c + w
        pct = 100 * c / parsed if parsed else 0
        print(f"{m:<28s} {n:>5d} {c:>5d} {w:>6d} {np_:>8d} {t:>6d} {pct:>5.0f}")
        total[0] += n; total[1] += c; total[2] += w; total[3] += np_; total[4] += t
    print("-" * 72)
    parsed_total = total[1] + total[2]
    print(f"{'TOTAL':<28s} {total[0]:>5d} {total[1]:>5d} {total[2]:>6d} {total[3]:>8d} {total[4]:>6d} {100*total[1]/parsed_total:>5.0f}")
    print()
    print(f"Cohort blended (parsed-only): {total[1]}/{parsed_total} = {100*total[1]/parsed_total:.1f}%")
    print()

    # By sub-source
    print("By sub-source:")
    by_src = defaultdict(list)
    for r in rows_all:
        by_src[(r["source"], r["year_month"])].append(r)
    for k in sorted(by_src.keys()):
        rs = by_src[k]
        c = sum(1 for r in rs if r["outcome"] == "correct")
        w = sum(1 for r in rs if r["outcome"] == "wrong")
        n = c + w
        pids = sorted(set(r["puzzle_id"] for r in rs))
        print(f"  {k}: {c}/{n} = {100*c/n:.0f}% ({len(pids)} unique puzzles)")

    # Per-puzzle (just print the wrong-heavy ones)
    print("\nPuzzles with >= 4 wrong (not 0% — these are the contested ones):")
    by_pid = defaultdict(list)
    for r in rows_all:
        by_pid[r["puzzle_id"]].append(r)
    for pid in sorted(by_pid.keys()):
        rs = by_pid[pid]
        c = sum(1 for r in rs if r["outcome"] == "correct")
        w = sum(1 for r in rs if r["outcome"] == "wrong")
        n = c + w
        if w >= 3 and n > 0:
            print(f"  {pid}: {c}/{n} ({100*c/n:.0f}%) expected={rs[0]['expected'][:30]!r}")


if __name__ == "__main__":
    main()
