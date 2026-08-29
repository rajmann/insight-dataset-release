"""Step 2 audit: build a blinded, stratified gold-standard sample.

Pulls traces from the deployed judge output (eureka_judge_full.csv for Rebus +
Cryptic, eureka_judge_vp.csv for VP) plus the source thinking traces from the
4-domain parquet, and writes two files:

  data/sample.json    - BLINDED data for the labelling tool. Per trace: domain,
                        model, puzzle_id, pass_type, chosen_answer, candidate
                        list (de-anchored shuffle), and the truncated trace text.
                        NO judge scores.
  data/gold_key.json   - HIDDEN key. Per (trace, candidate): the deployed judge's
                        stored judge_confidence + judge_ngrams + is_chosen. Loaded
                        ONLY by _analyse.py, never by the tool.

Stratification: per domain, traces are binned by the *chosen* candidate's stored
judge_confidence into fixed bins (the score piles up at 90-100, so equal qcut is
impossible) and the scarce low-confidence bin is oversampled because those
"I give up, going with X" traces are the most diagnostic. Selected traces are
emitted round-robin across (bin x domain) so that stopping early still yields a
balanced sample.

VP is restricted to the un_augmented pass for consistency with Rebus/Cryptic
(the deployed VP judge actually ran on both passes).

Usage:
  python _build_sample.py [--per-domain 25] [--seed 42]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # repo root
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PARQUET = ROOT / "llm_evaluation" / "datasets" / "insight_4domain" / "all_traces.parquet"

# fixed confidence bins on the chosen candidate's stored judge score
BIN_EDGES = [0, 60, 90, 95, 101]
BIN_LABELS = ["low", "mid", "high", "eureka"]
TRACE_MAX_CHARS = 20000   # must match _eureka_llm_judge.truncate_trace


def truncate_trace(trace: str, max_chars: int = TRACE_MAX_CHARS) -> str:
    """Verbatim copy of the deployed judge's truncation so the tool shows the
    exact text the judge saw."""
    if not isinstance(trace, str):
        return ""
    if len(trace) <= max_chars:
        return trace
    head = max_chars // 2
    tail = max_chars - head
    return trace[:head] + "\n\n[... middle elided for length ...]\n\n" + trace[-tail:]


def build_trace_table(judge_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Collapse one-row-per-candidate judge output into one row per trace with a
    clean chosen-candidate confidence and an ordered, de-duplicated candidate
    list. Drops traces that don't have exactly one chosen candidate."""
    rows = []
    for key, g in judge_df.groupby(key_cols, sort=False):
        g = g.reset_index(drop=True)
        chosen = g[g["is_chosen"] == 1]
        if len(chosen) != 1:
            continue
        chosen_conf = chosen["judge_confidence"].iloc[0]
        if pd.isna(chosen_conf):
            continue
        # ordered, de-duplicated candidate list (file order = input order)
        cands, seen = [], set()
        for c in g["candidate"].tolist():
            if not isinstance(c, str) or not c.strip():
                continue
            if c in seen:
                continue
            seen.add(c)
            cands.append(c)
        if len(cands) < 2:
            continue
        rec = dict(zip(key_cols, key if isinstance(key, tuple) else (key,)))
        rec.update(
            chosen_conf=float(chosen_conf),
            chosen_answer=chosen["chosen_answer"].iloc[0],
            candidates=cands,
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def stratified_pick(trace_tbl: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Pick n traces from one domain, spread across the fixed confidence bins,
    oversampling the scarce low bin. Returns the picks with a 'bin' column."""
    t = trace_tbl.copy()
    t["bin"] = pd.cut(t["chosen_conf"], bins=BIN_EDGES, labels=BIN_LABELS,
                      right=False, include_lowest=True)
    target = {b: n // len(BIN_LABELS) for b in BIN_LABELS}
    for i in range(n - sum(target.values())):          # spread remainder low->high
        target[BIN_LABELS[i % len(BIN_LABELS)]] += 1

    picks, shortfall = [], 0
    avail = {b: t[t["bin"] == b] for b in BIN_LABELS}
    for b in BIN_LABELS:
        want = target[b]
        pool = avail[b]
        take = min(want, len(pool))
        if take:
            picks.append(pool.sample(take, random_state=int(rng.integers(1e9))))
        shortfall += want - take
    # redistribute shortfall (mostly from the low bin) onto bins with surplus
    if shortfall:
        chosen_idx = pd.concat(picks).index if picks else pd.Index([])
        leftover = t.drop(index=chosen_idx)
        # prefer high/eureka surplus
        leftover = leftover.sort_values("chosen_conf", ascending=False)
        extra = leftover.head(shortfall)
        if len(extra):
            picks.append(extra)
    out = pd.concat(picks).drop_duplicates(subset=["model", "puzzle_id"] +
                                           (["pass_type"] if "pass_type" in t else []))
    return out.head(n)


def shuffle_candidates(cands: list[str], trace_id: str) -> list[str]:
    """Deterministic per-trace shuffle to de-anchor display order."""
    seed = abs(hash(trace_id)) % (2**32)
    r = np.random.default_rng(seed)
    idx = r.permutation(len(cands))
    return [cands[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading judge output + parquet ...")
    jf = pd.read_csv(ROOT / "eureka_judge_full.csv")            # Rebus + Cryptic
    vp = pd.read_csv(ROOT / "eureka_judge_vp.csv")
    vp = vp[vp["pass_type"] == "un_augmented"].copy()           # consistency
    parquet = pd.read_parquet(PARQUET)

    # trace lookup keyed by (model, domain, puzzle_id, pass_type)
    trace_lookup = {}
    for _, r in parquet.iterrows():
        trace_lookup[(r["model"], r["domain"], str(r["puzzle_id"]), r["pass_type"])] = r["thinking"] or ""

    # judge_full has no pass_type -> it's un_augmented (per _eureka_llm_judge.py)
    jf["pass_type"] = "un_augmented"

    domain_frames = {
        "Rebus": jf[jf["domain"] == "Rebus"],
        "Cryptic": jf[jf["domain"] == "Cryptic"],
        "VP": vp,
    }

    selected = {}   # domain -> list of trace records (dicts)
    for dom, df in domain_frames.items():
        tbl = build_trace_table(df, ["model", "domain", "puzzle_id", "pass_type"])
        picks = stratified_pick(tbl, args.per_domain, rng)
        recs = []
        for _, row in picks.iterrows():
            recs.append(row.to_dict())
        # report bin coverage
        binct = picks["bin"].value_counts().reindex(BIN_LABELS, fill_value=0).to_dict()
        print(f"  {dom:8s}: {len(picks):3d} traces from {len(tbl)} eligible | bins {binct}")
        selected[dom] = recs

    # round-robin interleave across (domain, bin) so partial labelling stays balanced
    # group each domain's recs by bin, then emit in rounds
    by_cell = {}
    for dom, recs in selected.items():
        for rec in recs:
            b = str(pd.cut([rec["chosen_conf"]], bins=BIN_EDGES, labels=BIN_LABELS,
                           right=False, include_lowest=True)[0])
            by_cell.setdefault((b, dom), []).append(rec)
    cell_order = [(b, d) for b in BIN_LABELS for d in domain_frames]
    ordered = []
    round_i = 0
    while True:
        added = False
        for cell in cell_order:
            lst = by_cell.get(cell, [])
            if round_i < len(lst):
                ordered.append(lst[round_i])
                added = True
        if not added:
            break
        round_i += 1

    # build blinded sample + hidden key
    sample_traces, gold_key = [], {}
    n_no_trace = 0
    for rec in ordered:
        model, dom, pid, ptype = rec["model"], rec["domain"], str(rec["puzzle_id"]), rec["pass_type"]
        trace_id = f"{dom}|{model}|{pid}|{ptype}"
        raw = trace_lookup.get((model, dom, pid, ptype), "")
        if not raw:
            n_no_trace += 1
            continue
        cands = rec["candidates"]
        disp = shuffle_candidates(cands, trace_id)
        sample_traces.append({
            "trace_id": trace_id,
            "domain": dom,
            "model": model,
            "puzzle_id": pid,
            "pass_type": ptype,
            "candidates": disp,
            "trace_text": truncate_trace(raw),
        })
        # gold key: per-candidate stored judge score
        src = domain_frames[dom]
        sub = src[(src["model"] == model) & (src["puzzle_id"].astype(str) == pid) &
                  (src["pass_type"] == ptype)]
        cand_map = {}
        for _, cr in sub.iterrows():
            c = cr["candidate"]
            if not isinstance(c, str) or c in cand_map:
                continue
            cand_map[c] = {
                "judge_confidence": (None if pd.isna(cr["judge_confidence"])
                                     else float(cr["judge_confidence"])),
                "judge_ngrams": cr.get("judge_ngrams", "[]"),
                "is_chosen": int(cr["is_chosen"]),
            }
        gold_key[trace_id] = {
            "chosen_conf": rec["chosen_conf"],
            "candidates": cand_map,
        }

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "sample.json").write_text(json.dumps(
        {"meta": {"per_domain": args.per_domain, "seed": args.seed,
                  "n_traces": len(sample_traces), "trace_max_chars": TRACE_MAX_CHARS,
                  "bin_edges": BIN_EDGES, "bin_labels": BIN_LABELS},
         "traces": sample_traces}, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "gold_key.json").write_text(json.dumps(gold_key, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    n_pairs = sum(len(t["candidates"]) for t in sample_traces)
    print(f"\nwrote {DATA/'sample.json'}  ({len(sample_traces)} traces, ~{n_pairs} candidate pairs)")
    print(f"wrote {DATA/'gold_key.json'} (hidden key)")
    if n_no_trace:
        print(f"NOTE: {n_no_trace} selected traces dropped (no thinking text in parquet)")


if __name__ == "__main__":
    main()
