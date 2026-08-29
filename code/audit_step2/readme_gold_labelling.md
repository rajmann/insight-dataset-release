# Gold-standard labelling - quick start

Blinded self-annotation to validate the per-candidate confidence judge (Step 2).
You re-score the traces the judge saw, blind to its scores and to the emitted
answer, then compare via Spearman rho.

## Run it

All commands from the repo root (`scripts phd\rebus`), venv active:

```powershell
# 1. Build the blinded sample (already done - only rerun to change size/seed)
python "llm_evaluation\audit_step2\_build_sample.py"

# 2. Start the labelling UI
python "llm_evaluation\audit_step2\server.py"
#    -> open http://localhost:8765 in a browser. Ctrl-C to stop.

# 3. Check progress / results any time (works on whatever you've scored so far)
python "llm_evaluation\audit_step2\_analyse.py"
```

Or `cd "llm_evaluation\audit_step2"` first and drop the path prefix.

## How to score

For each trace you see the full thinking trace + its candidate list. For every
candidate, type a **0-100** confidence score based only on the model's language
(not whether it's right). The rubric is pinned at the bottom of the page.

- **Click a candidate** (its name or score box) -> all its mentions light up in
  the trace.
- **Click words in the trace** -> they build an n-gram for the selected
  candidate. Tag it `+ supports` or `- opposes`, then **Add** (or drag-select
  text and use **+ selection**). Click a chip to delete it.
- N-grams are optional but capture **both** supporting and opposing phrases.
- **Committing is not confidence**: "I give up, going with X" scores LOW (0-30).

Everything autosaves to `data\annotations.json` after each change. You can stop
any time and resume later; traces are interleaved across domains and confidence
bins, so a partial pass stays balanced.

## Files

- `_build_sample.py` - builds `data\sample.json` (blinded) + `data\gold_key.json` (hidden judge scores)
- `server.py` / `index.html` - the labelling tool
- `_analyse.py` - Spearman (pooled + per-domain), bootstrap CIs, quartile kappa, disagreement examples
- `README.md` - fuller design notes

Blind by design: the emitted answer and the judge's scores are never shown.
