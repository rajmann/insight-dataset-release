# Step 2 audit: judge gold-standard labelling

Blinded self-annotation to validate the per-candidate confidence judge
(`_eureka_llm_judge.py`). I re-score the same traces the deployed judge saw,
blind to its scores AND to the emitted answer, then compare via Spearman rho.

## Pipeline

```
python _build_sample.py            # -> data/sample.json (blinded) + data/gold_key.json (hidden)
python server.py                   # open http://localhost:8765 and score
python _analyse.py                 # -> Spearman/kappa/CIs + data/analysis_results.json
```

## Files

| File | Role |
|------|------|
| `_build_sample.py` | 75-trace stratified sample (25/domain). Bins by the chosen candidate's stored judge score (`<60 / 60-89 / 90-94 / 95-100`, low oversampled). VP = un_augmented only. Trace text truncated to 20 000 chars exactly as the deployed judge. Candidate order shuffled per trace to de-anchor. |
| `server.py` | Python stdlib server. Serves the blinded sample + UI, persists `data/annotations.json` on every change. Never serves the gold key. |
| `index.html` | Labelling UI. Trace pane (clickable words) + per-candidate score box (0-100) and n-gram box. Click a candidate -> highlight its mentions; click trace words -> build an n-gram, tag + supports / - opposes, Add. Rubric pinned at the bottom. Autosaves. |
| `_analyse.py` | Spearman (pooled + per-domain), Pearson, cluster-bootstrap 95% CI (clusters = domain x puzzle_id), quartile quadratic-weighted Cohen's kappa, judge>=80 / <=20 disagreement bands, top-8 disagreements with both sides' n-grams. |

## Design notes

- **Blind always**: the emitted answer is not in `sample.json` and not shown. The
  deployed judge never knew which candidate was committed; neither do you.
- **Per-trace scoring**: you see the whole candidate list for a trace at once,
  matching the judge's actual prompt input (it scored all candidates jointly).
- **N-grams**: both supporting and opposing phrases, per the deployed prompt.
- Stop early any time: traces are interleaved across (bin x domain), so a partial
  pass stays balanced. `_analyse.py` works on whatever is scored so far.
