"""Step 4: cross-family second judge (DeepSeek V3 via OpenRouter).

Re-scores the 75 gold-standard traces with a DIFFERENT model family, using the
deployed judge's exact prompt and the original (un-shuffled) trace + candidate
order. Tests whether the per-candidate confidence ranking is an artifact of one
model (Gemini 3 Flash) or holds across families. This is the cross-family
robustness control reviewers expect; it expands the Appendix E pilot (Llama 3.3,
rho = 0.74) with a stronger, independent judge.

DeepSeek is chosen because it is independent of every model UNDER evaluation
(Gemini, Claude, GPT, Qwen all appear in the judged set) and is capable enough
that a disagreement cannot be dismissed as a weak judge.

Reports DeepSeek vs the deployed Gemini judge (cross-family agreement) and
DeepSeek vs the human gold standard, pooled and split chosen / non-chosen.

Usage:
  python _audit_step4_second_judge.py [--model deepseek/deepseek-chat] [--workers 4]
  python _audit_step4_second_judge.py --compare-only data/second_judge_deepseek.json
Env: OPENROUTER_API_KEY (loaded from .env, same as the openrouter_* scripts).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from _eureka_llm_judge import PROMPT_TEMPLATE                 # exact deployed prompt
from _audit_step3b_shuffle import candidate_order_by_trace    # reuse input-order reconstruction

from dotenv import load_dotenv
for _envp in (HERE.parent / ".env", ROOT / ".env"):
    if _envp.exists():
        load_dotenv(_envp)


def call_openrouter(prompt: str, model: str, temperature: float = 0.0):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],   # text-only: the judge never sees the image
        max_tokens=4000,
        temperature=temperature,
    )
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None, text
    try:
        return json.loads(m.group(0)), text
    except json.JSONDecodeError:
        return None, text


def generate(model: str, workers: int, limit: int | None) -> Path:
    sample = json.loads((DATA / "sample.json").read_text(encoding="utf-8"))["traces"]
    if limit:
        sample = sample[:limit]
    order = candidate_order_by_trace()

    jobs = []
    for t in sample:
        cands = order.get(t["trace_id"]) or t["candidates"]
        jobs.append((t["trace_id"], cands, t["trace_text"]))
    print(f"judging {len(jobs)} traces with {model}")

    def one(job):
        tid, cands, trace = job
        prompt = PROMPT_TEMPLATE.format(
            candidate_list="\n".join(f"- {c}" for c in cands), trace=trace)
        try:
            parsed, raw = call_openrouter(prompt, model)
        except Exception as e:
            return tid, None, f"call_failed: {e}"
        if not parsed or "candidates" not in parsed:
            return tid, None, f"parse_failed: {str(raw)[:120]!r}"
        scores = {o.get("candidate"): o.get("confidence")
                  for o in parsed["candidates"] if isinstance(o.get("candidate"), str)}
        return tid, scores, None

    results, errors, done = {}, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for fut in as_completed(futs):
            tid, scores, err = fut.result()
            done += 1
            if err:
                errors += 1
                print(f"  [{done}/{len(jobs)}] {tid} {err[:80]}")
            else:
                results[tid] = {"scores": scores}
            if done % 20 == 0:
                print(f"  [{done}/{len(jobs)}] done, {errors} errors")

    slug = model.replace("/", "_")
    out_path = DATA / f"second_judge_{slug}.json"
    out_path.write_text(json.dumps(
        {"meta": {"model": model, "n_traces": len(results)}, "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}  ({len(results)} traces, {errors} errors)")
    return out_path


def _rho(rows, x, y):
    rs = [r for r in rows if r[x] is not None and r[y] is not None]
    if len(rs) < 5:
        return None, len(rs)
    return spearmanr([r[x] for r in rs], [r[y] for r in rs]).statistic, len(rs)


def compare(path: Path):
    sj = json.loads(path.read_text(encoding="utf-8"))
    model = sj["meta"]["model"]; sj = sj["results"]
    key = json.loads((DATA / "gold_key.json").read_text(encoding="utf-8"))
    ann = json.loads((DATA / "annotations.json").read_text(encoding="utf-8")) \
        if (DATA / "annotations.json").exists() else {}

    rows = []
    for tid, r in sj.items():
        ck = key.get(tid, {}).get("candidates", {})
        human = ann.get(tid, {}).get("scores") or {}
        for c, dconf in r["scores"].items():
            rows.append(dict(
                deepseek=None if dconf is None else float(dconf),
                gemini=ck.get(c, {}).get("judge_confidence"),
                human=human.get(c),
                chosen=ck.get(c, {}).get("is_chosen", 0)))

    print(f"\n=== Cross-family second judge: {model} ===")
    chosen = [r for r in rows if r["chosen"]]
    nonch = [r for r in rows if not r["chosen"]]
    for label, sub in [("ALL pairs", rows), ("chosen only", chosen), ("non-chosen", nonch)]:
        rg, ng = _rho(sub, "deepseek", "gemini")
        rh, nh = _rho(sub, "deepseek", "human")
        rg = "n/a" if rg is None else round(rg, 3)
        rh = "n/a" if rh is None else round(rh, 3)
        print(f"  {label:12s}  vs Gemini judge: rho={rg} (n={ng})   vs human: rho={rh} (n={nh})")
    # for reference: Gemini-vs-human on the same pairs (the Step 2 headline)
    rgh, n = _rho(chosen, "gemini", "human")
    print(f"\n  reference: Gemini-vs-human on chosen = {None if rgh is None else round(rgh,3)} (n={n})  [Step 2 headline]")
    print("  (cross-family agreement near the Gemini-vs-human level => ranking is not a single-model artifact)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-chat",
                    help="OpenRouter slug; default is DeepSeek V3 chat. Override for a specific version.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="judge only the first N traces (smoke test)")
    ap.add_argument("--compare-only", default=None)
    args = ap.parse_args()

    if args.compare_only:
        compare(Path(args.compare_only)); return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Set OPENROUTER_API_KEY (loaded from .env, like the openrouter_* scripts).")
    out = generate(args.model, args.workers, args.limit)
    compare(out)


if __name__ == "__main__":
    main()
