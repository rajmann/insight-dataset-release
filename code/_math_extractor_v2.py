"""Fixed math answer extractor.

Three targeted fixes (no over-correction):

  Fix 1: \\text{...} preserves CONTENT instead of being stripped entirely.
         Old: \\text{(E)} -> "" (empty); was a normaliser bug.
         New: \\text{(E)} -> (E)
         Then optionally strip outer parens if they wrap a single letter
         token like (E) -> E (multiple-choice convention).

  Fix 2: trailing _<digits> subscript stripping (base-N answers).
         Old: '52_8' != '52'.
         New: strip _<digits>$ from both. Accept only if at most one had
         an explicit suffix, OR both had the same suffix (so e.g. _8 vs _9
         is still rejected even after stripping).

  Fix 3: comma-separated answers are unordered SETS when not wrapped in
         parens.
         Old: '1,-2' != '-2,1'.
         New: if no surrounding parens, sort the comma-separated parts
         before comparison. Coordinate-style '(0,1)' vs '(1,0)' is still
         distinguished (parens preserved -> ordered).

Re-scores all 700 math un_aug rows and reports:
  - Number of now-correct cases (was wrong, now correct)
  - Number of now-wrong cases (was correct, now wrong; should be 0)
  - List of newly-correct cases for manual audit
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "math_100" / "results_un_aug"
sys.path.insert(0, str(HERE))
from _math_canonical_lr import is_correct as is_correct_v1, MODELS


def normalise_math_v2(s):
    """Improved math answer normaliser with the three fixes."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^\$+|\$+$", "", s)

    # Strip \boxed{...} wrapper, keep inner content
    m = re.match(r"^\\boxed\{(.*)\}$", s)
    if m:
        s = m.group(1)

    # \frac{a}{b} -> a/b
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
    # \frac a b -> a/b (no-brace form, multi-digit, whitespace-separated)
    s = re.sub(r"\\[dt]?frac\s+(-?\d+)\s+(-?\d+)", r"\1/\2", s)
    # \frac 59 -> 5/9 (LaTeX single-digit shorthand)
    s = re.sub(r"\\[dt]?frac\s+(\d)(\d)\b", r"\1/\2", s)
    # \frac9{19} mixed form
    s = re.sub(r"\\[dt]?frac\s*(-?\d+)\{(-?\d+)\}", r"\1/\2", s)
    s = re.sub(r"\\[dt]?frac\{(-?\d+)\}\s*(-?\d+)", r"\1/\2", s)

    # FIX 1: \text{...} preserves content (was stripping entirely)
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)

    # Other common LaTeX strippers
    s = re.sub(r"\\left|\\right|\\,|\\!|\\;|\\:", "", s)
    s = re.sub(r"\^\\?circ|\^{\\?circ}", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)
    s = s.lower()

    # FIX 1b: strip outer parens around single letter (multiple-choice (E) -> E)
    m = re.match(r"^\(([a-z])\)$", s)
    if m:
        s = m.group(1)

    return s


def get_base_suffix(s):
    """Extract trailing _<digits> base subscript, return (stripped_s, suffix_or_None)."""
    m = re.search(r"^(.*)_(\d+)$", s)
    if m:
        return m.group(1), m.group(2)
    return s, None


UNIT_SUFFIX_RE = re.compile(
    r"(?<=\d)(cm|mm|km|inches?|in|ft|feet|°|degrees?|deg|rad|radians?|"
    r"seconds?|minutes?|hours?|kg|g|lbs?|oz|%|percent)$",
    re.IGNORECASE,
)


def strip_unit_suffix(s):
    """Strip a common unit suffix that follows a digit, e.g. '12cm' -> '12'.
    The lookbehind ensures we only strip after a digit, so variable-like
    suffixes (e.g. '12m' meaning 12*m) are not naively stripped."""
    return UNIT_SUFFIX_RE.sub("", s)


def is_correct_v2(answer, expected):
    """Fixed correctness check. Returns True only if answer is mathematically
    equivalent to expected, allowing for format/ordering differences but NOT
    for genuine value differences."""
    if answer is None:
        return False
    a = normalise_math_v2(answer)
    e = normalise_math_v2(expected)
    if not a or not e:
        return False

    # Direct match after normalisation
    if a == e:
        return True

    # FIX 4: unit suffix handling (e.g. '12cm' vs '12'). Both must strip to
    # the same prefix. If both have units they must match.
    a_no_unit = strip_unit_suffix(a)
    e_no_unit = strip_unit_suffix(e)
    a_unit_m = UNIT_SUFFIX_RE.search(a)
    e_unit_m = UNIT_SUFFIX_RE.search(e)
    if a_no_unit == e_no_unit and a_no_unit:
        if (a_unit_m is None) or (e_unit_m is None) or (a_unit_m.group(0).lower() == e_unit_m.group(0).lower()):
            return True

    # FIX 2: base-N suffix handling
    # Strip _<digits> suffix if either has one. Accept ONLY if base prefixes
    # match AND the suffixes (when both present) are equal.
    a_pre, a_sfx = get_base_suffix(a)
    e_pre, e_sfx = get_base_suffix(e)
    if a_pre == e_pre:
        # Same prefix; suffixes must be compatible
        if (a_sfx is None) or (e_sfx is None) or (a_sfx == e_sfx):
            return True

    # FIX 3: comma-separated unordered set match (only if no surrounding parens
    # in EITHER, since parens-wrapped answers like (1,2) are coordinates and
    # are ordered)
    def has_outer_parens(x):
        return x.startswith("(") and x.endswith(")")

    if "," in a and "," in e and not has_outer_parens(a) and not has_outer_parens(e):
        a_parts = sorted(p.strip() for p in a.split(","))
        e_parts = sorted(p.strip() for p in e.split(","))
        if a_parts == e_parts:
            return True

    # Numeric equality fallback (existing logic)
    def to_float(x):
        try:
            if "/" in x:
                num, den = x.split("/", 1)
                return float(num) / float(den)
            return float(x)
        except (ValueError, ZeroDivisionError):
            return None
    fa, fe = to_float(a), to_float(e)
    if fa is not None and fe is not None and abs(fa - fe) < 1e-9:
        return True

    return False


def main():
    rows = []
    for model_name, fname in MODELS:
        path = DATA_DIR / f"{fname}.jsonl"
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            ans = r.get("answer_reparsed") or r.get("answer")
            expected = r.get("expected", "")
            v1 = int(is_correct_v1(ans, expected))
            v2 = int(is_correct_v2(ans, expected))
            rows.append({
                "model": model_name,
                "puzzle_id": r["puzzle_id"],
                "subject": r.get("subject", ""),
                "level": r.get("level"),
                "expected": expected,
                "answer": ans,
                "correct_v1": v1,
                "correct_v2": v2,
            })

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"Total math rows: {n}")
    print(f"v1 accuracy: {df['correct_v1'].mean():.3f} (n_wrong={int((1-df['correct_v1']).sum())})")
    print(f"v2 accuracy: {df['correct_v2'].mean():.3f} (n_wrong={int((1-df['correct_v2']).sum())})")
    print()

    now_correct = df[(df["correct_v1"] == 0) & (df["correct_v2"] == 1)]
    now_wrong = df[(df["correct_v1"] == 1) & (df["correct_v2"] == 0)]
    print(f"Now correct (was wrong, fixed): {len(now_correct)}")
    print(f"Now wrong (was correct, broke): {len(now_wrong)} (should be 0)")
    print()

    print("Audit table: newly-correct cases (was wrong, now correct)")
    print(f"  {'Model':<24s} {'Puzzle':<12s} {'Expected':<22s} {'Model answer':<22s}")
    for _, r in now_correct.iterrows():
        exp = (r["expected"] or "")[:20]
        ans = (r["answer"] or "")[:20]
        print(f"  {r['model']:<24s} {r['puzzle_id']:<12s} {exp!r:<22s} {ans!r:<22s}")

    if len(now_wrong) > 0:
        print()
        print("WARNING: cases where v1 said correct but v2 says wrong:")
        for _, r in now_wrong.iterrows():
            print(f"  {r['model']:<24s} {r['puzzle_id']:<12s} expected={r['expected']!r} answer={r['answer']!r}")

    # Save the v2-scored rows for downstream use
    out_path = HERE / "math_100" / "math_v2_scored.jsonl"
    df.to_json(out_path, orient="records", lines=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
