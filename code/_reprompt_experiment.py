"""Re-prompt experiment (INLG revision, downstream-utility E1).

Take pass-1 (un_augmented) traces, flag the lowest-confidence quartile per
(model, domain) using the DEPLOYABLE no-SR classifier, and re-query the SAME model
with: the original pass-1 prompt + image + its own pass-1 thinking trace + its pass-1
answer + a confidence-score cue, asking it to (a) explore directions it did not try and
(b) reconsider what it did, then emit NEW ANSWER + PREFERRED ANSWER.

Design is locked in readme_inlg_revision_plan.md. Variant A only (score cue).

Reuse notes (do not reinvent):
- Flagging classifier: predictions.parquet, config='a', feature_subset='canonical_no_sr',
  cv_setup='lopo_within_domain', pass_type='un_augmented' -> predicted_prob (verified: one
  clean row per model-domain-puzzle).
- Provider wrappers + resume/concurrency patterns mirror _visualpuzzles_full_eval.py and
  _connections_plus_inference.py.
- Original pass-1 prompts are the un_augmented templates from _build_paper_dataset.py.

This script COLLECTS pass-2 outputs only. Correctness scoring / flip analysis is a
separate step (_reprompt_analyse.py) so it can reuse the exact paper normalisers with
answer_alternatives.

Usage:
  python _reprompt_experiment.py --dry-run                 # build prompts + flag, no API
  python _reprompt_experiment.py --models "Gemini 3 Flash,Gemini 3 Pro" \
        --domains Rebus,Cryptic --workers 6                 # pilot (~$8)
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent                 # .../llm_evaluation/datasets
LLM = HERE.parent                                      # .../llm_evaluation
load_dotenv(LLM / ".env")

PAPER = HERE / "paper_dataset_4domain_2026-05-05"
OUT_DIR = Path(os.environ.get("REPROMPT_DIR") or (HERE / "reprompt_results"))
OUT_DIR.mkdir(exist_ok=True)

# (display_name, provider, model_id) -- lifted verbatim from the per-domain scripts.
# NOTE: gemini-3-pro-preview (the pass-1 snapshot) was RETIRED for inference post-collection
# (listed but 404s on generate). We substitute the successor gemini-3.1-pro-preview and
# ASTERISK Gemini 3 Pro results: a newer model reconsidering an older model's trace. The
# headline utility claim rests on the 6 models whose exact pass-1 snapshot is still live.
SUBSTITUTED_SNAPSHOT = {"Gemini 3 Pro": "gemini-3-pro-preview (retired) -> gemini-3.1-pro-preview"}
MODELS = {
    "Gemini 3 Pro":            ("ai_studio",  "gemini-3.1-pro-preview"),
    "Gemini 3 Flash":          ("ai_studio",  "gemini-3-flash-preview"),
    "Gemini 2.5 Pro":          ("ai_studio",  "gemini-2.5-pro"),
    "GPT-5":                   ("openrouter", "openai/gpt-5"),
    "Claude Sonnet 4.6":       ("anthropic",  "claude-sonnet-4-6"),
    "Claude Opus 4.6":         ("anthropic",  "claude-opus-4-6"),
    "Qwen3-VL-235B Thinking":  ("openrouter", "qwen/qwen3-vl-235b-a22b-thinking"),
}
# trace parquet uses this exact model string; note the Qwen suffix.
MODEL_ALIASES = {"Qwen3-VL-235B": "Qwen3-VL-235B Thinking"}

DOMAINS = ["Rebus", "Cryptic", "VP", "ConnectionsPlus"]
# max_output / thinking_budget per domain for the re-prompt pass. Hard bottom-quartile
# puzzles ruminate heavily (~15k thinking tokens observed), so max_output must sit well
# above thinking so the NEW/PREFERRED lines are not truncated.
BUDGETS = {"Rebus": (32000, 16000), "Cryptic": (32000, 16000),
           "VP": (16000, 8000), "ConnectionsPlus": (32000, 16000)}

_file_lock: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()


def lock_for(path: Path) -> threading.Lock:
    with _lock_guard:
        return _file_lock.setdefault(str(path), threading.Lock())


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


# ---------------------------------------------------------------- flagging
def load_flagged(models: list[str], domains: list[str], quantile: float) -> pd.DataFrame:
    """Lowest-confidence `quantile` per (model, domain), joined to pass-1 trace text."""
    preds = pd.read_parquet(PAPER / "predictions.parquet")
    preds["pass_type"] = preds["row_id"].str.rsplit("|", n=1).str[-1]
    sub = preds[(preds["config"] == "a") &
                (preds["feature_subset"] == "canonical_no_sr") &
                (preds["cv_setup"] == "lopo_within_domain") &
                (preds["pass_type"] == "un_augmented")].copy()
    sub = sub[sub["model"].isin(models) & sub["domain"].isin(domains)]
    # bottom quantile within each (model, domain)
    keep = []
    for (m, d), g in sub.groupby(["model", "domain"]):
        thr = g["predicted_prob"].quantile(quantile)
        keep.append(g[g["predicted_prob"] <= thr])
    flagged = pd.concat(keep, ignore_index=True) if keep else sub.iloc[0:0]

    traces = pd.read_parquet(PAPER / "all_traces.parquet")
    tcols = ["row_id", "thinking", "answer", "expected", "correct", "raw_output"]
    flagged = flagged.merge(traces[tcols], on="row_id", how="left")
    # drop rows with no usable pass-1 trace to feed back
    flagged = flagged[flagged["thinking"].fillna("").str.len() > 0].reset_index(drop=True)
    return flagged


# ---------------------------------------------------------------- domain inputs
_cryptic_cache: dict | None = None


def cryptic_index() -> dict:
    global _cryptic_cache
    if _cryptic_cache is None:
        idx = json.loads((HERE / "cryptic_100" / "index.json").read_text(encoding="utf-8"))
        by_id = {}
        for row in idx["clues"]:
            by_id[row["puzzle_id"]] = dict(row)
        # enumeration lives in ground_truth
        for pid, row in by_id.items():
            gt_p = HERE / "cryptic_100" / "ground_truth" / f"{pid}.json"
            if gt_p.exists():
                gt = json.loads(gt_p.read_text(encoding="utf-8"))
                row["enumeration"] = gt.get("enumeration", "")
        _cryptic_cache = by_id
    return _cryptic_cache


def puzzle_input(domain: str, puzzle_id: str):
    """Return (text_context, image_bytes_or_None) reconstructing the pass-1 stimulus."""
    if domain == "Rebus":
        img = (LLM / "images" / f"{puzzle_id}.png").read_bytes()
        return "", img
    if domain == "Cryptic":
        row = cryptic_index()[puzzle_id]
        return f"Clue: {row['clue']}\nEnumeration: {row.get('enumeration', '')}", None
    raise NotImplementedError(f"input loader for {domain} not wired yet (VP/ConnP: full run)")


# ---------------------------------------------------------------- prompts
# Original un_augmented framing (verbatim from _build_paper_dataset.py), minus the
# "respond with ONLY / do NOT explain" line since here we WANT reconsideration.
DOMAIN_FRAMING = {
    "Rebus": "This is a rebus puzzle image. What word or phrase does it represent?",
    "Cryptic": "Solve this cryptic crossword clue.",
    "VP": ("This is a visual analogy puzzle. Find the relationship between Origin and "
           "Analogy, then apply it to Origin' to determine the correct Analogy' among the "
           "four candidates A, B, C, D."),
}
ANSWER_LINE = {
    "Rebus": ("NEW ANSWER: <your best alternative answer, or \"none\" if you cannot find a "
              "plausible one>\nPREFERRED ANSWER: <either your original answer or your new answer>"),
    "Cryptic": ("NEW ANSWER: <your best alternative answer in UPPERCASE, or \"none\">\n"
                "PREFERRED ANSWER: <either your original answer or your new answer, UPPERCASE>"),
    "VP": ("NEW ANSWER: <one letter A, B, C, or D, or \"none\">\n"
           "PREFERRED ANSWER: <one letter A, B, C, or D>"),
}


def build_reprompt(domain: str, text_context: str, pass1_thinking: str,
                   pass1_answer: str, conf_pct: int) -> str:
    framing = DOMAIN_FRAMING[domain]
    ctx = f"\n{text_context}\n" if text_context else "\n"
    return (
        f"{framing}\n{ctx}"
        f"You attempted this puzzle earlier. This was your reasoning:\n"
        f"\"\"\"\n{pass1_thinking.strip()}\n\"\"\"\n\n"
        f"Your earlier answer was: {clean_answer(pass1_answer)}.\n"
        f"However, this answer was scored as having only {conf_pct}% confidence.\n\n"
        f"Take a fresh look. First, consider directions or interpretations your earlier "
        f"reasoning did not explore, and try to find a genuinely different candidate answer "
        f"that might fit better. Then re-examine the ideas you did have. Finally, decide "
        f"which single answer you prefer, between your original answer and your new one.\n\n"
        f"Respond with:\n{ANSWER_LINE[domain]}"
    )


def clean_answer(ans: str) -> str:
    ans = (ans or "").strip()
    ans = re.sub(r"^\s*ANSWER\s*:\s*", "", ans, flags=re.I)
    return ans.strip()


# ---------------------------------------------------------------- output parsing
def parse_reprompt(text: str) -> tuple[str, str]:
    """Return (new_answer, preferred_answer). '' if a field is missing."""
    def grab(label):
        m = re.search(rf"{label}\s*:\s*(.+)", text, flags=re.I)
        return m.group(1).strip() if m else ""
    new = grab("NEW ANSWER")
    pref = grab("PREFERRED ANSWER")
    return new, pref


# ---------------------------------------------------------------- providers
def call_gemini(provider: str, model_id: str, prompt: str, image: bytes | None,
                max_out: int, think_budget: int) -> dict:
    from google import genai
    from google.genai import types
    if provider == "vertex":
        client = genai.Client(vertexai=True, project=os.environ.get("GCP_PROJECT"), location="us-central1")
    else:
        key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_GENAI_API_KEY"]
        client = genai.Client(api_key=key)
    contents = []
    if image is not None:
        contents.append(types.Part.from_bytes(data=image, mime_type="image/png"))
    contents.append(prompt)
    resp = client.models.generate_content(
        model=model_id, contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=max_out,
            thinking_config=types.ThinkingConfig(include_thoughts=True,
                                                 thinking_budget=think_budget)))
    thinking, output = [], []
    for part in resp.candidates[0].content.parts:
        if getattr(part, "text", None) is None:
            continue
        (thinking if getattr(part, "thought", False) else output).append(part.text)
    u = resp.usage_metadata
    return {"raw_output": "".join(output), "thinking": "".join(thinking),
            "tokens_thinking": getattr(u, "thoughts_token_count", 0) or 0,
            "tokens_output": getattr(u, "candidates_token_count", 0) or 0}


def call_anthropic(model_id: str, prompt: str, image: bytes | None,
                   max_out: int, think_budget: int) -> dict:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content = []
    if image is not None:
        content.append({"type": "image", "source": {"type": "base64",
                        "media_type": "image/png", "data": base64.b64encode(image).decode()}})
    content.append({"type": "text", "text": prompt})
    with client.messages.stream(model=model_id, max_tokens=max_out,
                                thinking={"type": "enabled", "budget_tokens": think_budget},
                                messages=[{"role": "user", "content": content}]) as stream:
        resp = stream.get_final_message()
    thinking, output = [], []
    for block in resp.content:
        if block.type == "thinking":
            thinking.append(block.thinking)
        elif block.type == "text":
            output.append(block.text)
    return {"raw_output": "".join(output), "thinking": "".join(thinking),
            "tokens_thinking": 0, "tokens_output": resp.usage.output_tokens}


def call_openrouter(model_id: str, prompt: str, image: bytes | None, max_out: int) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1", timeout=480.0, max_retries=0)
    if image is not None:
        content = [{"type": "image_url", "image_url":
                    {"url": f"data:image/png;base64,{base64.b64encode(image).decode()}"}},
                   {"type": "text", "text": prompt}]
    else:
        content = prompt
    resp = client.chat.completions.create(model=model_id, max_tokens=max_out,
                                          extra_body={"include_reasoning": True},
                                          messages=[{"role": "user", "content": content}])
    msg = resp.model_dump()["choices"][0]["message"]
    return {"raw_output": (msg.get("content") or "").strip(),
            "thinking": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "tokens_thinking": 0, "tokens_output": 0}


def call_openai_responses(model_id: str, prompt: str, max_out: int) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.responses.create(
        model=model_id,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        reasoning={"effort": "medium", "summary": "detailed"}, max_output_tokens=max_out)
    raw = resp.model_dump()
    thinking, output = [], []
    for item in raw["output"]:
        if item["type"] == "reasoning":
            thinking += [s["text"] for s in item.get("summary", [])]
        elif item["type"] == "message":
            output += [c["text"] for c in item["content"] if c.get("type") == "output_text"]
    u = raw.get("usage", {})
    return {"raw_output": "".join(output), "thinking": "".join(thinking),
            "tokens_thinking": u.get("output_tokens_details", {}).get("reasoning_tokens", 0),
            "tokens_output": u.get("output_tokens", 0)}


def dispatch(model_name: str, domain: str, prompt: str, image: bytes | None) -> dict:
    provider, model_id = MODELS[model_name]
    max_out, tb = BUDGETS[domain]
    if provider in ("ai_studio", "vertex"):
        return call_gemini(provider, model_id, prompt, image, max_out, tb)
    if provider == "anthropic":
        return call_anthropic(model_id, prompt, image, max_out, tb)
    if provider == "openrouter":
        return call_openrouter(model_id, prompt, image, max_out)
    raise ValueError(provider)


# ---------------------------------------------------------------- runner
def run_model(model_name: str, rows: pd.DataFrame, workers: int, dry: bool):
    ckpt = OUT_DIR / f"{safe_name(model_name)}.jsonl"
    done = set()
    if ckpt.exists() and not dry:
        # Keep only GOOD rows (non-error, non-empty PREFERRED); rewrite file so truncated /
        # errored rows are purged and get retried on this run. Dedupe by (domain, puzzle_id).
        good = {}
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "error" in rec or not rec.get("preferred_answer"):
                continue
            good[(rec["domain"], rec["puzzle_id"])] = rec
        with open(ckpt, "w", encoding="utf-8") as fh:
            for rec in good.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done = set(good.keys())
    elif ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["domain"], r["puzzle_id"]))
            except Exception:
                pass
    todo = [r for _, r in rows.iterrows() if (r["domain"], r["puzzle_id"]) not in done]
    print(f"[{model_name}] {len(todo)} to do ({len(done)} already done)")
    lock = lock_for(ckpt)

    def work(r):
        domain, pid = r["domain"], r["puzzle_id"]
        conf_pct = int(round(float(r["predicted_prob"]) * 100))
        try:
            text_ctx, image = puzzle_input(domain, pid)
            prompt = build_reprompt(domain, text_ctx, r["thinking"], r["answer"], conf_pct)
        except Exception as e:  # input/prompt build failure
            return {"model": model_name, "domain": domain, "puzzle_id": pid,
                    "error": f"input: {e}"}
        if dry:
            return {"model": model_name, "domain": domain, "puzzle_id": pid,
                    "conf_pct": conf_pct, "prompt_chars": len(prompt),
                    "pass1_answer": clean_answer(r["answer"]), "pass1_correct": int(r["correct"]),
                    "prompt_preview": prompt[:600]}
        for attempt in range(3):
            try:
                t0 = time.time()
                out = dispatch(model_name, domain, prompt, image)
                elapsed = time.time() - t0
                new, pref = parse_reprompt(out["raw_output"])
                return {"model": model_name, "model_id": MODELS[model_name][1],
                        "snapshot_note": SUBSTITUTED_SNAPSHOT.get(model_name, ""),
                        "domain": domain, "puzzle_id": pid,
                        "predicted_prob": float(r["predicted_prob"]), "conf_pct": conf_pct,
                        "pass1_answer": clean_answer(r["answer"]),
                        "pass1_correct": int(r["correct"]), "expected": r["expected"],
                        "new_answer": new, "preferred_answer": pref,
                        "reprompt_raw": out["raw_output"], "reprompt_thinking": out["thinking"],
                        "tokens_thinking": out["tokens_thinking"],
                        "tokens_output": out["tokens_output"],
                        "elapsed_seconds": elapsed, "variant": "A"}
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "rate", "overloaded")):
                    time.sleep(15 * 2 ** attempt)
                else:
                    time.sleep(3)
                if attempt == 2:
                    return {"model": model_name, "domain": domain, "puzzle_id": pid,
                            "error": msg[:300]}

    def write(rec):
        if rec is None:
            return
        with lock:
            with open(ckpt, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    if dry or workers == 1:
        for r in todo:
            write(work(r))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for rec in ex.map(work, todo):
                write(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Gemini 3 Flash,Gemini 3 Pro")
    ap.add_argument("--domains", default="Rebus,Cryptic")
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap re-prompts per model (0=all); for smoke test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = [MODEL_ALIASES.get(m.strip(), m.strip()) for m in args.models.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    for m in models:
        assert m in MODELS, f"unknown model {m!r}"

    flagged = load_flagged(models, domains, args.quantile)
    print("Flagged rows:", len(flagged))
    print(flagged.groupby(["model", "domain"]).size().to_string())
    if args.dry_run:
        print("\n--- DRY RUN (no API calls) ---")
    for m in models:                       # one model at a time, sequential
        rows = flagged[flagged["model"] == m]
        if args.limit:
            rows = rows.head(args.limit)
        if len(rows):
            run_model(m, rows, args.workers, args.dry_run)


if __name__ == "__main__":
    main()
