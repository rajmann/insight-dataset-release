"""Stochasticity stability check (INLG revision; supervisor control on single-run pairings).

The paper pairs each correctness label with trace features from ONE generation per puzzle,
and reasoning models are stochastic (several do not even accept a temperature). This script
re-generates an INDEPENDENT run of the ORIGINAL un_augmented pass for a sample of puzzles so
the paper's correct-vs-wrong feature contrast can be re-measured on a fresh run.

Run 1 = the existing traces in all_traces.parquet. This script collects run >= 2 and, with
--analyse, reports per-run Cohen's d (wrong minus correct) for the six headline features plus
per-run accuracy and the correct<->wrong flip rate. Temperature is NOT set: the stochasticity
we test is the one that actually occurs at our generation settings.

Reuses (no reinvention):
  - provider caller + clue index from _reprompt_experiment (call_gemini, cryptic_index)
  - feature extractor from _extract_features (extract_for_row; the 6 features use no trajectory)
  - correctness from _reprompt_analyse (gold_set, is_correct)
  - un_augmented prompt verbatim from _build_paper_dataset

Pilot: --model "Gemini 3 Flash" --domain Cryptic --run run2  (all Flash Cryptic puzzles).

Usage:
  python _stability_run.py --model "Gemini 3 Flash" --domain Cryptic --run run2 --workers 8
  python _stability_run.py --model "Gemini 3 Flash" --domain Cryptic --analyse
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _reprompt_experiment import (call_gemini, call_anthropic, call_openrouter,
                                  cryptic_index, MODELS)
from _extract_features import extract_for_row
from _reprompt_analyse import gold_set, is_correct

PAPER = HERE / "paper_dataset_4domain_2026-05-05"
OUT_DIR = HERE / "stability_results"
OUT_DIR.mkdir(exist_ok=True)

FEATS = ["tokens_thinking_proxy", "elapsed", "thinking_char_count",
         "hedge_rate", "hedge_ratio", "hedge_position_variance"]

# verbatim un_augmented prompt (from _build_paper_dataset.PROMPTS)
CRYPTIC_PROMPT = ("Solve this cryptic crossword clue.\n\n"
                  "Clue: {clue}\n"
                  "Enumeration: {enumeration}\n\n"
                  "Do NOT explain. Respond with ONLY this exact syntax:\n"
                  "ANSWER: <answer in UPPERCASE; use a single space between words for multi-word answers>")
# (max_output, thinking_budget) per model. Gemini/Claude Cryptic traces are ~2-3k chars,
# so 32k/16k never truncates. Qwen3-VL runs to ~58k chars (p95); give the openrouter cap
# plenty of room so long traces are not clipped (which would bias the length features).
def budget_for(model: str):
    return (64000, 24000) if "Qwen" in model else (32000, 16000)


def dispatch_call(model: str, prompt: str, image, max_out: int, tb: int) -> dict:
    provider, model_id = MODELS[model]
    if provider in ("ai_studio", "vertex"):
        return call_gemini(provider, model_id, prompt, image, max_out, tb)
    if provider == "anthropic":
        return call_anthropic(model_id, prompt, image, max_out, tb)
    if provider == "openrouter":
        return call_openrouter(model_id, prompt, image, max_out)
    raise ValueError(provider)

_lock = threading.Lock()


def safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")


def extract_answer(raw: str) -> str:
    m = re.search(r"ANSWER\s*:\s*(.+)", raw or "", flags=re.I)
    return (m.group(1).strip() if m else (raw or "").strip())


def run1_rows(model: str, domain: str) -> pd.DataFrame:
    at = pd.read_parquet(PAPER / "all_traces.parquet")
    r = at[(at.model == model) & (at.domain == domain) & (at.pass_type == "un_augmented")]
    return r[["puzzle_id", "expected"]].drop_duplicates("puzzle_id").reset_index(drop=True)


# ------------------------------------------------------------------ generation
def generate(model: str, domain: str, run: str, workers: int, limit: int):
    assert domain == "Cryptic", "wired for Cryptic (text-only); add image loader for others"
    model_id = MODELS[model][1]
    max_out, tb = budget_for(model)
    idx = cryptic_index()
    rows = run1_rows(model, domain)
    if limit:
        rows = rows.head(limit)

    ckpt = OUT_DIR / f"{safe(model)}__{domain}__{run}.jsonl"
    done = set()
    if ckpt.exists():
        for ln in ckpt.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(ln)
                if "error" not in rec and rec.get("raw_output"):
                    done.add(rec["puzzle_id"])
            except Exception:
                pass
    todo = [r for _, r in rows.iterrows() if r["puzzle_id"] not in done]
    print(f"[{model} | {domain} | {run}] {len(todo)} to do ({len(done)} done)")

    def work(r):
        pid = r["puzzle_id"]
        clue = idx.get(pid, {})
        prompt = CRYPTIC_PROMPT.format(clue=clue.get("clue", ""),
                                       enumeration=clue.get("enumeration", ""))
        for attempt in range(3):
            try:
                t0 = time.time()
                out = dispatch_call(model, prompt, None, max_out, tb)
                el = time.time() - t0
                return {"model": model, "model_id": model_id, "domain": domain, "run": run,
                        "puzzle_id": pid, "expected": r["expected"],
                        "answer": extract_answer(out["raw_output"]),
                        "raw_output": out["raw_output"], "thinking": out["thinking"],
                        "tokens_thinking": out["tokens_thinking"],
                        "tokens_output": out["tokens_output"], "elapsed_seconds": el}
            except Exception as e:
                msg = str(e)
                time.sleep(15 * 2 ** attempt if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "overloaded")) else 3)
                if attempt == 2:
                    return {"model": model, "domain": domain, "run": run, "puzzle_id": pid,
                            "error": msg[:300]}

    def write(rec):
        if rec is None:
            return
        with _lock, open(ckpt, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if workers == 1:
        for r in todo:
            write(work(r))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(work, todo):
                write(rec)
    print(f"  wrote {ckpt}")


# ------------------------------------------------------------------ analysis
def cohens_d(sub, feat):
    c = sub[sub.correct == 1][feat].astype(float)
    w = sub[sub.correct == 0][feat].astype(float)
    if len(c) < 2 or len(w) < 2:
        return np.nan
    sp = np.sqrt(((len(c) - 1) * c.var(ddof=1) + (len(w) - 1) * w.var(ddof=1)) / (len(c) + len(w) - 2))
    return np.nan if sp == 0 else (w.mean() - c.mean()) / sp   # + => higher on WRONG


def feats_for_generation(recs, model, domain) -> pd.DataFrame:
    out = []
    for r in recs:
        row = {"thinking": r.get("thinking", "") or "", "answer": r.get("answer", "") or "",
               "tokens_thinking": int(r.get("tokens_thinking") or 0),
               "tokens_output": int(r.get("tokens_output") or 0),
               "elapsed_seconds": float(r.get("elapsed_seconds") or 0.0),
               "model": model, "domain": domain, "puzzle_id": r["puzzle_id"],
               "pass_type": "un_augmented", "self_reported_conf": None}
        f = extract_for_row(row, {}, {})
        golds = gold_set(domain, r["puzzle_id"], r.get("expected", ""))
        rec = {k: f[k] for k in FEATS}
        rec.update(puzzle_id=r["puzzle_id"], correct=int(is_correct(r.get("answer", ""), golds)))
        out.append(rec)
    return pd.DataFrame(out)


def run1_frame(model, domain) -> pd.DataFrame:
    """Run-1 features from features.parquet; correctness re-scored with is_correct for parity."""
    at = pd.read_parquet(PAPER / "all_traces.parquet")
    at = at[(at.model == model) & (at.domain == domain) & (at.pass_type == "un_augmented")]
    fe = pd.read_parquet(PAPER / "features.parquet")
    m = at.merge(fe[["row_id"] + FEATS], on="row_id", how="inner")
    rows = []
    for _, r in m.iterrows():
        golds = gold_set(domain, r["puzzle_id"], r.get("expected", "") or "")
        rec = {k: r[k] for k in FEATS}
        rec.update(puzzle_id=r["puzzle_id"], correct=int(is_correct(r.get("answer", "") or "", golds)))
        rows.append(rec)
    return pd.DataFrame(rows)


def analyse(model, domain):
    frames = {"run1": run1_frame(model, domain)}
    for ck in sorted(OUT_DIR.glob(f"{safe(model)}__{domain}__run*.jsonl")):
        run = ck.stem.split("__")[-1]
        recs = []
        for l in ck.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if "error" in r or not r.get("raw_output"):   # error KEY, not substring in trace
                continue
            recs.append(r)
        frames[run] = feats_for_generation(recs, model, domain)

    print(f"\n===== stability: {model} / {domain} =====")
    print(f"{'run':<6} {'n':>4} {'nCorr':>6} {'nWrong':>7} {'acc':>6}")
    for run, df in frames.items():
        print(f"{run:<6} {len(df):>4} {int(df.correct.sum()):>6} {int((df.correct==0).sum()):>7} {df.correct.mean():>6.2f}")

    runs = list(frames)
    fresh = [r for r in runs if r in ("run2", "run3")]      # same-snapshot pair (generated now)
    print(f"\nCohen's d (wrong - correct); + => feature HIGHER on wrong (matches -ve rho)")
    print(f"{'feature':<26} " + " ".join(f"{r:>8}" for r in runs)
          + f" {'rng(all)':>9} {'rng(2,3)':>9}")
    for f in FEATS:
        ds = {r: cohens_d(frames[r], f) for r in runs}
        vals = [v for v in ds.values() if not np.isnan(v)]
        rng = (max(vals) - min(vals)) if len(vals) > 1 else np.nan
        fv = [ds[r] for r in fresh if not np.isnan(ds[r])]
        rng23 = (max(fv) - min(fv)) if len(fv) > 1 else np.nan
        print(f"{f:<26} " + " ".join(f"{ds[r]:>8.2f}" if not np.isnan(ds[r]) else f"{'na':>8}" for r in runs)
              + f" {rng:>9.2f} {rng23:>9.2f}")

    def flip(x, y):
        a = frames[x].set_index("puzzle_id").correct
        b = frames[y].set_index("puzzle_id").correct
        common = a.index.intersection(b.index)
        return len(common), (a.loc[common] != b.loc[common]).mean()
    for x, y in [("run1", "run2"), ("run2", "run3")]:
        if x in frames and y in frames:
            n, fr = flip(x, y)
            tag = " (same-snapshot, clean)" if (x, y) == ("run2", "run3") else " (spans time gap)"
            print(f"{x} vs {y}: {n} shared, flip rate = {fr:.1%}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Gemini 3 Flash")
    ap.add_argument("--domain", default="Cryptic")
    ap.add_argument("--run", default="run2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        analyse(args.model, args.domain)
    else:
        generate(args.model, args.domain, args.run, args.workers, args.limit)


if __name__ == "__main__":
    main()
