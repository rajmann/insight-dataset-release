# Surface markers in reasoning traces — data and code

Reasoning traces, extracted features, and analysis code for the study of surface
effort/confidence markers in VLM reasoning traces on insight puzzles (Rebus, Cryptic
crosswords, VisualPuzzles, Connections+) and post-cutoff hard math.

Accompanies the INLG 2026 paper. Please cite the paper if you use this data or code.

## Layout

```
README.md            requirements.txt   LICENSE
data/
  insight_4domain/
    all_traces.parquet     # insight traces: thinking text, answers, outcomes, self-report
    features.parquet       # extracted surface features per trace
    predictions.parquet    # per-trace classifier predictions
    results/               # precomputed result JSONs (cascade, bootstrap CIs, …)
  hard_math_93/            # post-cutoff HMMT/AIME traces (un-augmented)
  connections_plus_100/    # OUR construction: 32-word grids + gold groupings + NYT dates
  reprompt_100/            # 100% re-prompt sweep (pass-2) for the reconsideration control
  stability_cryptic/       # two extra runs of the Cryptic pass for the single-run stability check
  provenance/              # puzzle_id -> external source refs + answers (see its README)
code/                      # analysis scripts (self-contained; import closure verified)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or your env of choice
pip install -r requirements.txt
cd code
python _math_appendix_spearman.py        # math-vs-insight Spearman (Tables 6, 13)
python _effort_feature_ablation.py       # single-feature + group ablation (Tables 14, 15)
python _difficulty_oracle_test.py        # classifier vs difficulty oracle (Table 16)
python _per_task_descriptive.py          # per-task descriptives (Table 17)
```

Each analysis script resolves its data via a path relative to the script, so it runs
from `code/` with no configuration. The core analyses are CPU-only and finish in
seconds.

## Reproduction map

| Paper element | Script | Notes |
|---|---|---|
| Tables 6, 13 — math vs insight Spearman | `code/_math_appendix_spearman.py` | reproduces exactly (n=335) |
| Table 13 note — Benjamini-Hochberg FDR | `code/_table12_fdr.py` | 27/27 starred correlations survive at q=0.05 |
| Tables 14, 15 — feature ablation | `code/_effort_feature_ablation.py` | reproduces exactly (full-6 mean AP 0.849) |
| Table 16 — difficulty oracle | `code/_difficulty_oracle_test.py` | reproduces exactly |
| Table 17 — per-task descriptives | `code/_per_task_descriptive.py` | reproduces exactly |
| Tables 2, 3, 11 — LODO / LOMO | `code/_4domain_effort_signal_classifier.py` | trains the effort-signal classifier. **Runs on all 5,054 extracted rows**, giving mean LODO AP 0.839 (Rebus 0.816 / Cryptic 0.906 / VP 0.842 / ConnP 0.792). Table 2 uses the judge-matched subset — see the row-set note under Data notes |
| Table 8 — cascade routing | `code/_cascade_pareto_reanalysis.py` | over `data/.../results/cascade_4domain.json` |
| Table 7, Table 18, escalation figure — reconsideration control | `code/_reprompt_control_ci.py` (with `_reprompt_analyse.py`, `_reprompt_control.py`, `_reprompt_curve_fig.py`) | targeting vs random vs top-quartile net gain over `data/reprompt_100/` |
| Limitations — single-run stability | `code/_stability_run.py --analyse` | per-feature Cohen's d across three runs over `data/stability_cryptic/` |
| App. — judge validation | `code/audit_step2/` | human/cross-family/position checks |
| Qwen within-model AUC | `code/_qwen_resolution_separability.py` | |

Collecting new traces (model re-runs) and the paper's auxiliary comparisons — the Table 4
embedding/BERT baselines and the Appendix token-level pilot — rely on external resources
(provider APIs, a GPU, a locally served Qwen3-8B) and are outside the scope of this bundle;
see the paper for those. The two collection harnesses `_reprompt_experiment.py` and
`_stability_run.py` (generation mode) likewise require provider APIs and the referenced
source puzzles; the shipped `data/reprompt_100/` and `data/stability_cryptic/` are their
outputs, and the analysis is reproducible from those.

## Data notes

- **Insight corpus**: 400 puzzles (100 per domain), attempted by 7 models over two passes
  (un-augmented + self-confidence). `all_traces.parquet` carries the full thinking text,
  the model's answer, the gold answer (`expected`), correctness, and self-reported
  confidence. Traces are shipped **verbatim**.
- **Math**: 50 post-cutoff HMMT/AIME problems, 348 traces (335 carry usable thinking used
  in the feature correlations).
- **Connections+** is our own construction (pairs of NYT Connections puzzles) and is shipped
  in full; every other domain is externally sourced and **referenced, not redistributed** —
  see `data/provenance/README.md`.
- **`reprompt_100/`**: pass-2 outputs of the 100% re-prompt sweep on Rebus and Cryptic. One
  JSONL per model; each record carries `model`, `puzzle_id`, the gold answer (`expected`),
  the pass-1 answer and correctness, the classifier confidence (`predicted_prob`), the new
  and preferred pass-2 answers, the pass-2 raw output and thinking, tokens, and elapsed time.
  Shipped verbatim; source clues remain referenced only. Gemini 3 Pro rows use a substituted
  snapshot (flagged in `snapshot_note`) and are excluded from the same-snapshot pooling.
- **`stability_cryptic/`**: two further independent runs (`run2`, `run3`) of the un-augmented
  Cryptic pass for three models, backing the single-run stability check. Each record carries
  `model`, `puzzle_id`, gold answer, answer, raw output, thinking, tokens, and `run`. Run 1
  is the corresponding un-augmented slice of `all_traces.parquet`.
- **`predictions.parquet`** holds held-out per-trace predictions (`predicted_prob`,
  `true_label`) over a grid of classifier variants: two CV setups — `lopo_within_domain`
  (leave-one-puzzle-out) and `fold5_domain_stratified` — crossed with 16 `feature_subset`
  ablations and `config` a–d (SR = self-reported-confidence feature; `no_sr` subsets use
  configs a/b, `with_sr` use c/d). **It does not contain LODO or LOMO predictions**: the
  cross-domain (Table 2) and cross-model (Table 3) results are recomputed live by
  `code/_4domain_effort_signal_classifier.py`; only the LOPO and 5-fold predictions are
  persisted here (these back the feature-ablation and within-domain numbers).
- **Row sets, and a known reproduction gap.** Table 2's two columns (`Effort AP` and `+conf AP`)
  are computed on the **judge-matched subset** — the 4,624 rows for which the LLM-judge feature
  `chosen_conf` exists — so that the two columns are paired on identical rows. That subset gives
  mean LODO AP 0.855. `_4domain_effort_signal_classifier.py` instead runs on all 5,054 extracted
  rows and gives 0.839. Both are correct for their row set; the effort classifier is identical.
  **`chosen_conf` is not currently shipped in this bundle**, so the `+conf` column and the exact
  Table 2 row set cannot be reproduced from these files. The judge scores will be added; until
  then, treat the shipped script's 0.839 as the all-rows figure, not a failed reproduction.

## Provenance and licensing

We do not redistribute the external source puzzles as datasets (Rebus/VisualPuzzles images,
Cryptic clue text, Math problem statements). `data/provenance/` maps each `puzzle_id` to its
original source with the ground-truth answer included. Note that the shipped reasoning traces
are model output and quote their source material incidentally: a trace may restate the clue or
describe the image it was reasoning about. The traces are unusable without this, and it is a
scattered subset rather than a redistribution of any source dataset. External datasets remain
under their own licences. The code and our derived data are released under the MIT `LICENSE`.
