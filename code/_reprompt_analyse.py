"""Score the re-prompt (E1) outputs and report the flip contingency.

Reads reprompt_results/*.jsonl, scores NEW / PREFERRED answers against gold
(expected + answer_alternatives) with the paper's textual normaliser, and reports
per-model + pooled: recovery (wrong->right), harm (right->wrong), net accuracy on the
flagged quartile, switch behaviour, and new-candidate analysis.

Self-check: re-scores pass1_answer and compares to the stored pass1_correct; a nonzero
mismatch count means the normaliser drifted from _build_paper_dataset.py.

Pilot = Rebus + Cryptic (textual scoring). VP/ConnP scoring stubbed for the full run.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
LLM = HERE.parent
OUT_DIR = Path(os.environ.get("REPROMPT_DIR") or (HERE / "reprompt_results"))

_GT: dict[tuple[str, str], list[str]] = {}


def gold_set(domain: str, puzzle_id: str, expected: str) -> list[str]:
    """Return normalisation-ready gold list: expected + answer_alternatives."""
    key = (domain, puzzle_id)
    if key in _GT:
        return _GT[key]
    golds = [expected] if expected else []
    p = None
    if domain == "Rebus":
        p = LLM / "ground_truth" / f"{puzzle_id}.json"
    elif domain == "Cryptic":
        p = HERE / "cryptic_100" / "ground_truth" / f"{puzzle_id}.json"
    if p and p.exists():
        gt = json.loads(p.read_text(encoding="utf-8"))
        if gt.get("answer"):
            golds.append(gt["answer"])
        golds += gt.get("answer_alternatives", []) or []
    _GT[key] = list({g for g in golds if g})
    return _GT[key]


def _norm(s: str) -> str:
    """Mirror _build_paper_dataset._norm_answer: lower, strip ANSWER:, drop non-alnum,
    drop articles, and->n, collapse ws."""
    s = (s or "").lower().strip()
    s = re.sub(r"^\s*answer\s*:\s*", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\band\b", "n", s)
    toks = [t for t in s.split() if t not in ("a", "an", "the")]
    return " ".join(toks).strip()


def is_correct(pred: str, golds: list[str]) -> bool:
    if not pred:
        return False
    np_ = _norm(pred)
    if not np_:
        return False
    np_ns = np_.replace(" ", "")
    for g in golds:
        ng = _norm(g)
        if np_ == ng or np_ns == ng.replace(" ", ""):
            return True
    return False


def load_records() -> list[dict]:
    recs = []
    for f in sorted(OUT_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" in r or "prompt_preview" in r:   # skip errors + dry-run rows
                continue
            if not (r.get("preferred_answer") or "").strip():  # skip truncated collections
                continue
            recs.append(r)
    return recs


def analyse(recs: list[dict], label: str):
    n = len(recs)
    if not n:
        print(f"\n[{label}] no records"); return
    mism = 0
    n_wrong = n_right = rec_recovered = rec_broke = 0
    switched = switched_correct = 0
    proposed_new = new_correct = 0
    pref_correct_total = 0
    p1_correct_total = 0
    for r in recs:
        golds = gold_set(r["domain"], r["puzzle_id"], r.get("expected", ""))
        p1 = int(r["pass1_correct"])
        p1_correct_total += p1
        if is_correct(r["pass1_answer"], golds) != bool(p1):
            mism += 1
        pref_c = is_correct(r.get("preferred_answer", ""), golds)
        pref_correct_total += int(pref_c)
        new_ans = (r.get("new_answer") or "").strip()
        pref_ans = (r.get("preferred_answer") or "").strip()
        # switch = preferred differs from pass-1 answer
        if pref_ans and _norm(pref_ans) != _norm(r["pass1_answer"]):
            switched += 1
            if pref_c:
                switched_correct += 1
        # new candidate proposed
        if new_ans and new_ans.lower() != "none" and _norm(new_ans) != _norm(r["pass1_answer"]):
            proposed_new += 1
            if is_correct(new_ans, golds):
                new_correct += 1
        if p1:
            n_right += 1
            if not pref_c:
                rec_broke += 1
        else:
            n_wrong += 1
            if pref_c:
                rec_recovered += 1
    p1_acc = p1_correct_total / n
    pref_acc = pref_correct_total / n
    print(f"\n===== {label}  (n={n}) =====")
    if mism:
        print(f"  !! normaliser self-check mismatch on {mism} pass-1 rows (expected ~0)")
    print(f"  pass-1 accuracy on flagged quartile : {p1_acc:.3f}  ({p1_correct_total}/{n})")
    print(f"  after re-prompt (PREFERRED)         : {pref_acc:.3f}  ({pref_correct_total}/{n})")
    print(f"  NET accuracy change                 : {pref_acc - p1_acc:+.3f}")
    print(f"  recovery  wrong->right : {rec_recovered}/{n_wrong}"
          f"  ({rec_recovered/n_wrong:.1%})" if n_wrong else "  recovery: n/a")
    print(f"  harm      right->wrong : {rec_broke}/{n_right}"
          f"  ({rec_broke/n_right:.1%})" if n_right else "  harm: n/a")
    print(f"  switched answer        : {switched}/{n}  (of which correct: {switched_correct})")
    print(f"  proposed a NEW candidate: {proposed_new}/{n}  (new answer correct: {new_correct})")


def main():
    recs = load_records()
    by_model: dict[str, list] = {}
    for r in recs:
        by_model.setdefault(r["model"], []).append(r)
    for m, rs in sorted(by_model.items()):
        note = rs[0].get("snapshot_note", "")
        analyse(rs, m + ("  [*snapshot substituted]" if note else ""))
    # pooled over same-snapshot models only (exclude substituted) + overall
    clean = [r for r in recs if not r.get("snapshot_note")]
    analyse(clean, "POOLED (same-snapshot models only)")
    analyse(recs, "POOLED (all, incl. substituted)")


if __name__ == "__main__":
    main()
