# Provenance

This directory maps each `puzzle_id` used in the release to its **original external
source**. Following our redistribution policy, we do **not** ship the source puzzle
bodies for externally-sourced domains (Rebus images, Cryptic clue text, Math problem
statements, VisualPuzzles images). Instead we provide precise references so the exact
items can be obtained from the original datasets, together with the **ground-truth
answers** (which we do include).

The one domain we constructed ourselves, **Connections+**, is shipped in full under
`../connections_plus_100/` (32-word grids + gold groupings + the two NYT source dates
per puzzle).

Note: the reasoning traces in `../insight_4domain/all_traces.parquet`
are our collected model outputs and are shipped verbatim. Where a trace quotes or
restates its source puzzle, that text remains in the trace unchanged — it is model
output, not a redistribution of the source dataset.

## Files

| File | Rows | `puzzle_id` maps to | Fields |
|---|---|---|---|
| `rebus.csv` | 100 | RE-BUS item id | `answer`, source ref |
| `visualpuzzles.csv` | 100 | neulab/VisualPuzzles analogical-subset **row index** | `hf_row_index`, `answer`, source ref |
| `cryptic.csv` | 100 | our `cc_###` id | `answer`, `publisher`, `sub_publisher`, `date`, `orientation`, `clue_number`, source |
| `math.csv` | 50 | competition problem id (e.g. `AIME-2026-1`) | `answer`, `source`, `year_month` |

## Sources and how to obtain the originals

- **Rebus** — RE-BUS (Das et al. 2025), arXiv:2511.01340, <https://rebus-dataset.github.io/>.
  `puzzle_id` (e.g. `10_3`) is the RE-BUS item id.
- **VisualPuzzles** — neulab/VisualPuzzles (Song et al. 2025),
  <https://huggingface.co/datasets/neulab/VisualPuzzles>. `puzzle_id` is the row index
  into the analogical subset; load the dataset and index directly.
- **Cryptic crosswords** — Cryptonite (Efrat et al. 2021),
  <https://github.com/aviaefrat/cryptonite>. The clue is identified by
  `publisher` + `date` + `orientation` + `clue_number`; look it up in the Cryptonite
  validation split.
- **Math** — public HMMT (November 2025, February 2026) and AIME 2026 competition
  problems (`source` + `year_month`). Problem statements are available from the
  respective competition archives.

Please cite the original datasets if you use these sources.
