"""Hard-math inference: 7 models on the 93-puzzle HMMT+AIME 2025-26 sample.

Mirrors `_math_inference.py` but with an olympiad-friendly prompt that does
not constrain answers to integers (HMMT 2026 has fractions, expressions with
pi/sqrt, etc.). Live extraction captures the last \\boxed{...} content as a
string; correctness grading is post-hoc via _math_extractor_v2.is_correct_v2.

Usage:
  python _hard_math_inference.py --pass-type un_aug
  python _hard_math_inference.py --pass-type aug
  # Pilot with --limit N or --puzzle-ids "AIME-2026-1,HMMT-2026-Feb-1"

Output:
  llm_evaluation/datasets/hard_math_93/results_<pass_type>/{Model}.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
SAMPLE_DIR = HERE / "hard_math_93"
MANIFEST = SAMPLE_DIR / "manifest.json"

sys.path.insert(0, str(HERE))
from _math_inference import MODELS, run_one_call, extract_confidence
from _math_inference import _extract_prose, _LAST_NUMBER  # for prose fallback


PROMPT_UN_AUG = """\
Solve the following math competition problem (HMMT or AIME). Think through it \
step-by-step. The answer may be an integer or a closed-form expression \
(e.g., a fraction like \\frac{{2}}{{5}}, an expression like 5\\pi + 6\\sqrt{{3}}, or a single integer).

You MUST put your final answer inside \\boxed{{}}, exactly as in this example: \
"The answer is \\boxed{{42}}." or "\\boxed{{\\frac{{1}}{{2}}}}". This format is required for grading.

Problem: {problem}
"""

PROMPT_AUG = """\
Solve the following math competition problem (HMMT or AIME). Think through it \
step-by-step. The answer may be an integer or a closed-form expression \
(e.g., a fraction like \\frac{{2}}{{5}}, an expression like 5\\pi + 6\\sqrt{{3}}, or a single integer).

You MUST put your final answer inside \\boxed{{}}, exactly as in this example: \
"The answer is \\boxed{{42}}." or "\\boxed{{\\frac{{1}}{{2}}}}". This format is required for grading.

After your boxed answer, on a separate line, write CONFIDENCE: followed by an \
integer between 0 and 100 indicating how confident you are that your answer is \
correct.

Problem: {problem}
"""


def extract_boxed(text: str):
    """Extract the last \\boxed{...} content with full brace-balance matching.

    Handles arbitrary nesting depth (e.g., \\boxed{\\frac{3\\sqrt{5}}{7}}).
    Scans for \\boxed{ openers and walks forward counting { and } until balance
    returns to zero. Returns the LAST balanced match. Falls back to prose
    extraction (last paragraph or "answer is N" patterns) for outputs that
    don't use \\boxed (e.g., Gemini 2.5 Pro often produces prose answers).
    """
    if not text:
        return None
    results = []
    i = 0
    pat = re.compile(r"\\boxed\{")
    while True:
        m = pat.search(text, i)
        if not m:
            break
        start = m.end()
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j-1].strip())
            i = j
        else:
            break
    if results:
        return results[-1]
    # Fallback: prose patterns from the math reparser (handles "the answer is X")
    prose = _extract_prose(text)
    if prose:
        return prose
    return None


def run_model(model_name, provider, model_id, items, prompt_template, out_dir, print_lock):
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    ckpt_path = out_dir / f"{safe_name}.jsonl"

    done_ids = set()
    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["puzzle_id"])
                except Exception:
                    pass
        with print_lock:
            print(f"[{model_name}] resuming, {len(done_ids)} already done")

    with open(ckpt_path, "a", encoding="utf-8") as fh:
        for idx, item in enumerate(items, 1):
            if item["puzzle_id"] in done_ids:
                continue
            prompt = prompt_template.format(problem=item["problem"])
            for attempt in range(3):
                try:
                    r = run_one_call(provider, model_id, prompt)
                    pred = extract_boxed(r["output"])
                    conf = extract_confidence(r["output"])
                    rec = {
                        "puzzle_id": item["puzzle_id"],
                        "source": item["source"],
                        "year_month": item["year_month"],
                        "problem_type": item.get("problem_type", ""),
                        "expected": item["answer"],
                        "answer": pred,
                        "self_reported_conf": conf,
                        "elapsed": r["elapsed"],
                        "thinking": r["thinking"],
                        "raw_output": r["output"],
                        "tokens_thinking": r["tokens_thinking"],
                        "tokens_output": r["tokens_output"],
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    with print_lock:
                        ok = "OK" if pred else "NO_ANS"
                        cnf = f"conf={conf}" if conf is not None else "no_conf"
                        print(f"[{model_name}] {idx}/{len(items)} "
                              f"{item['puzzle_id']} pred={(pred or '<none>')[:25]} {ok} {cnf} "
                              f"exp={item['answer'][:18]} ({r['elapsed']:.1f}s)")
                    break
                except Exception as e:
                    msg = str(e)[:200]
                    with print_lock:
                        print(f"[{model_name}] {idx}/{len(items)} "
                              f"{item['puzzle_id']} attempt {attempt+1} "
                              f"error: {type(e).__name__}: {msg}")
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate" in msg.lower():
                        time.sleep(15 * (2 ** attempt))
                    else:
                        time.sleep(3)
                    if attempt == 2:
                        rec = {
                            "puzzle_id": item["puzzle_id"],
                            "source": item["source"],
                            "year_month": item["year_month"],
                            "expected": item["answer"],
                            "error": f"{type(e).__name__}: {msg}",
                        }
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-type", choices=["un_aug", "aug"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=7)
    ap.add_argument("--puzzle-ids", type=str, default=None)
    args = ap.parse_args()

    print(f"Loading {MANIFEST}...")
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    items = manifest["puzzles"]
    items.sort(key=lambda x: x["puzzle_id"])

    if args.puzzle_ids:
        wanted = set(args.puzzle_ids.split(","))
        items = [p for p in items if p["puzzle_id"] in wanted]
    elif args.limit:
        items = items[:args.limit]

    out_dir = SAMPLE_DIR / f"results_{args.pass_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = PROMPT_UN_AUG if args.pass_type == "un_aug" else PROMPT_AUG

    print(f"\nPass type: {args.pass_type}")
    print(f"Output dir: {out_dir}")
    print(f"Total items: {len(items)}")
    print(f"Models: {[m[0] for m in MODELS]}")
    print()

    print_lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(run_model, name, prov, mid, items, prompt_template, out_dir, print_lock): name
                for name, prov, mid in MODELS}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
                with print_lock:
                    print(f"[{name}] DONE")
            except Exception as e:
                with print_lock:
                    print(f"[{name}] WORKER FAILED: {type(e).__name__}: {e}")
    elapsed = time.time() - t0
    print(f"\nAll workers finished in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
