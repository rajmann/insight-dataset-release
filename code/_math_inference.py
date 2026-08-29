"""Math-domain inference: 7 models on the 100-puzzle MATH-500 sample.

Mirrors the resume + parallel pattern from `_visualpuzzles_baseline_pass.py`:
  - One thread per model (ThreadPoolExecutor)
  - JSONL append-only with f.flush() per record
  - Resume on restart by loading already-done puzzle_ids from the JSONL
  - Exponential backoff on 429 / RESOURCE_EXHAUSTED
  - Error records persisted so hard failures don't loop indefinitely

Usage:
  # Pilot: 10 puzzles x 7 models, un-augmented pass (no CONFIDENCE)
  python _math_inference.py --pass-type un_aug --limit 10

  # Same 10 puzzles, augmented pass (asks for ANSWER + CONFIDENCE)
  python _math_inference.py --pass-type aug --limit 10

  # Continue to all 100 (skips already-done puzzle_ids automatically)
  python _math_inference.py --pass-type un_aug
  python _math_inference.py --pass-type aug

Output:
  llm_evaluation/datasets/math_100/results_<pass_type>/{Model}.jsonl   (one JSONL per model)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
HERE = Path(__file__).resolve().parent
# .env lives in llm_evaluation/, this script is in llm_evaluation/datasets/
load_dotenv(HERE.parent / ".env")
SAMPLE_DIR = HERE / "math_100"
MANIFEST = SAMPLE_DIR / "manifest.json"


# Full 7-model lineup. Claude uses the direct Anthropic SDK because
# OpenRouter silently drops Claude's extended-thinking content (see VP/ConnP
# inference scripts which discovered this earlier).
MODELS = [
    ("Gemini 3 Pro",              "ai_studio",   "gemini-3-pro-preview"),
    ("Gemini 3 Flash",            "ai_studio",   "gemini-3-flash-preview"),
    ("Gemini 2.5 Pro",            "vertex",      "gemini-2.5-pro"),
    ("GPT-5",                     "openrouter",  "openai/gpt-5"),
    ("Claude Sonnet 4.6",         "anthropic",   "claude-sonnet-4-6"),
    ("Claude Opus 4.6",           "anthropic",   "claude-opus-4-6"),
    ("Qwen3-VL-235B Thinking",    "openrouter",  "qwen/qwen3-vl-235b-a22b-thinking"),
]


# Prompt templates - text-only math, ask for boxed answer
PROMPT_UN_AUG = """\
Solve the following math problem. Think through it step-by-step. \
You MUST put your final answer inside \\boxed{{}}, exactly as in this example: \
"The answer is \\boxed{{42}}." This format is required for grading.

Problem: {problem}
"""

PROMPT_AUG = """\
Solve the following math problem. Think through it step-by-step. \
You MUST put your final answer inside \\boxed{{}}, exactly as in this example: \
"The answer is \\boxed{{42}}." This format is required for grading.

After your boxed answer, on a separate line, write CONFIDENCE: followed by an \
integer between 0 and 100 indicating how confident you are that your answer is \
correct.

Problem: {problem}
"""


# Mirrors the offline reparser in `_math_reparse.py` so live answer-extraction
# matches the post-hoc reparse pass (recovers Gemini 2.5 Pro prose answers).
_ANS_TOKEN = (
    r"-?\d+(?:[./]\d+)?"
    r"|\\[a-zA-Z]+(?:\{[^{}]+\}){0,4}"
    r"|\([^()]{1,30}\)"
)
_PROSE_PATTERNS = [
    re.compile(rf"\b(?:the\s+)?answer\s+is\s+({_ANS_TOKEN})", re.IGNORECASE),
    re.compile(rf"\b(?:probability|value|result|minimum|maximum|sum|number|count|"
               rf"length|distance|total|product|quotient|difference)"
               rf"(?:\s+\w+){{0,5}}\s+(?:is|=|equals)\s+({_ANS_TOKEN})", re.IGNORECASE),
    re.compile(rf"\b(?:Therefore|Thus|Hence|So)[,]?\s+(?:there\s+(?:is|are)\s+)?"
               rf"(?:the\s+\S+\s+(?:is|=)\s+)?({_ANS_TOKEN})", re.IGNORECASE),
    re.compile(rf"=\s*({_ANS_TOKEN})\s*\.?\s*$", re.MULTILINE),
]
_FINAL_IS = re.compile(rf"\b(?:is|equals|=)\s+({_ANS_TOKEN})\b", re.IGNORECASE)
_LAST_NUMBER = re.compile(rf"(?:^|[\s>=:])({_ANS_TOKEN})\b\s*\.?\s*$", re.MULTILINE)


def _cleanup(s):
    if not s:
        return s
    return re.sub(r"[\s\.,;:`\)\]]+$", "", s.strip()).strip()


def _extract_prose(text):
    text = text.replace("`", "").replace("$", "").rstrip()
    last_para = text.rsplit("\n\n", 1)[-1]
    if last_para and last_para != text and len(last_para) <= 600:
        cands = []
        for pat in _PROSE_PATTERNS + [_FINAL_IS]:
            for m in pat.finditer(last_para):
                cand = _cleanup(m.group(1))
                if cand and len(cand) <= 60:
                    cands.append((m.end(), cand))
        if cands:
            cands.sort(key=lambda x: x[0])
            return cands[-1][1]
        m = _LAST_NUMBER.search(last_para)
        if m:
            return _cleanup(m.group(1))
    tail = text[-500:]
    cands = []
    for pat in _PROSE_PATTERNS:
        for m in pat.finditer(tail):
            cand = _cleanup(m.group(1))
            if cand and len(cand) <= 60:
                cands.append((m.end(), cand))
    if cands:
        cands.sort(key=lambda x: x[0])
        return cands[-1][1]
    m = _LAST_NUMBER.search(tail)
    if m:
        return _cleanup(m.group(1))
    return None


def extract_answer(text: str):
    """Extract the answer from the model's output.

    Hierarchy:
      1. Last \\boxed{...} (preferred; what the prompt asks for)
      2. Last LaTeX inline expression in the last 200 chars
      3. Prose patterns (catches Gemini 2.5 Pro etc. when they ignore \\boxed)
      4. None
    """
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if matches:
        return matches[-1].strip()
    a = _extract_prose(text)
    if a:
        return a
    tail = text[-200:]
    inline = re.findall(r"\$([^$]+)\$", tail)
    if inline:
        return inline[-1].strip()
    return None


def extract_confidence(text: str):
    """Extract CONFIDENCE: <0-100> from the augmented-pass output."""
    if not text:
        return None
    m = re.search(r"CONFIDENCE\s*:\s*(\d+)", text)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 100:
                return v
        except ValueError:
            pass
    return None


# ---- Provider call wrappers (mirrors visualpuzzles_baseline_pass conventions) ----

def _genai_client(provider):
    from google import genai
    if provider == "vertex":
        return genai.Client(vertexai=True, project=os.environ.get("GCP_PROJECT"),
                            location="us-central1")
    if provider == "ai_studio":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_GENAI_API_KEY")
        return genai.Client(api_key=key)
    raise ValueError(f"unknown provider: {provider}")


def call_gemini(provider, model_id, prompt):
    from google.genai import types
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    client = _genai_client(provider)
    t0 = time.time()
    # Per-call timeout to escape Gemini API hangs that block the worker thread
    # indefinitely on certain hard math problems. 6 minutes is more than enough
    # for genuine completion (longest observed: ~3 min).
    def _do_call():
        return client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                max_output_tokens=32000,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True, thinking_budget=16000),
            ),
        )
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_do_call)
        try:
            resp = fut.result(timeout=360)
        except FuturesTimeoutError:
            raise TimeoutError(f"Gemini call timed out after 360s on {model_id}")
    elapsed = time.time() - t0
    thoughts, outs = [], []
    if resp.candidates:
        for p in resp.candidates[0].content.parts:
            t = getattr(p, "text", "")
            if not t:
                continue
            (thoughts if getattr(p, "thought", False) else outs).append(t)
    usage = getattr(resp, "usage_metadata", None)
    return {
        "thinking": "\n".join(thoughts),
        "output": "\n".join(outs),
        "elapsed": elapsed,
        "tokens_thinking": getattr(usage, "thoughts_token_count", 0) or 0,
        "tokens_output": getattr(usage, "candidates_token_count", 0) or 0,
    }


def call_openrouter(model_id, prompt):
    from openai import OpenAI
    # Per-call timeout (480s = 8 min) escapes OpenRouter requests that hang
    # on certain hard math problems with the larger token budget. GPT-5 was
    # observed hung indefinitely on 60+ min idle in earlier pilots.
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                    timeout=480.0,
                    max_retries=0)
    t0 = time.time()
    # GPT-5 / Qwen3-VL Thinking can produce 30-60K thinking tokens on hard
    # olympiad problems; keep this generous so the visible answer doesn't
    # get truncated. Cost goes up roughly linearly with response length.
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=64000,
        extra_body={"include_reasoning": True},
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.time() - t0
    raw = resp.model_dump()
    msg = raw["choices"][0]["message"]
    return {
        "thinking": msg.get("reasoning") or msg.get("reasoning_content") or "",
        "output": (msg.get("content") or "").strip(),
        "elapsed": elapsed,
        "tokens_thinking": 0,
        "tokens_output": 0,
    }


def call_anthropic(model_id, prompt):
    """Direct Anthropic SDK with extended thinking enabled.

    Mirrors `_connections_plus_inference.py:call_anthropic`. Captures the
    `thinking` content blocks separately from visible answer text. Streaming is
    used because non-streaming requests > 10 min raise ValueError.
    """
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    t0 = time.time()
    with client.messages.stream(
        model=model_id,
        max_tokens=48000,
        thinking={"type": "enabled", "budget_tokens": 24000},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        resp = stream.get_final_message()
    elapsed = time.time() - t0
    thinking_parts, output_parts = [], []
    for block in resp.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            output_parts.append(block.text)
    return {
        "thinking": "\n".join(thinking_parts),
        "output": "\n".join(output_parts).strip(),
        "elapsed": elapsed,
        "tokens_thinking": 0,
        "tokens_output": getattr(resp.usage, "output_tokens", 0) or 0,
    }


def run_one_call(provider, model_id, prompt):
    if provider in ("vertex", "ai_studio"):
        return call_gemini(provider, model_id, prompt)
    if provider == "anthropic":
        return call_anthropic(model_id, prompt)
    return call_openrouter(model_id, prompt)


# ---- Main worker per model ----

def run_model(model_name, provider, model_id, items, prompt_template, out_dir, print_lock):
    """Process all puzzles for one model. Resume-safe; appends to JSONL."""
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_")
    ckpt_path = out_dir / f"{safe_name}.jsonl"

    # Resume: load already-done puzzle_ids
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
                    pred = extract_answer(r["output"])
                    conf = extract_confidence(r["output"])
                    rec = {
                        "puzzle_id": item["puzzle_id"],
                        "subject": item["subject"],
                        "level": item["level"],
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
                        mark = "OK" if pred else "NO_ANSWER"
                        cnf = f"conf={conf}" if conf is not None else "no_conf"
                        print(f"[{model_name}] {idx}/{len(items)} "
                              f"{item['puzzle_id']} pred={(pred or '<none>')[:30]} {mark} {cnf} "
                              f"({r['elapsed']:.1f}s)")
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
                            "subject": item["subject"],
                            "level": item["level"],
                            "expected": item["answer"],
                            "error": f"{type(e).__name__}: {msg}",
                        }
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-type", choices=["un_aug", "aug"], required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Restrict to first N puzzles (sorted by puzzle_id). Useful for pilot.")
    ap.add_argument("--max-workers", type=int, default=7,
                    help="Concurrent workers (one per model by default).")
    args = ap.parse_args()

    print(f"Loading {MANIFEST}...")
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    items = manifest["puzzles"]
    items.sort(key=lambda x: x["puzzle_id"])
    print(f"  total puzzles in manifest: {len(items)}")

    if args.limit:
        items = items[:args.limit]
        print(f"  limited to first {len(items)} puzzles (pilot mode)")
        print(f"  puzzle_ids: {[p['puzzle_id'] for p in items]}")

    out_dir = SAMPLE_DIR / f"results_{args.pass_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = PROMPT_UN_AUG if args.pass_type == "un_aug" else PROMPT_AUG

    print(f"\nPass type: {args.pass_type}")
    print(f"Output dir: {out_dir}")
    print(f"Models: {[m[0] for m in MODELS]}")
    print(f"Workers: {args.max_workers}")
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
