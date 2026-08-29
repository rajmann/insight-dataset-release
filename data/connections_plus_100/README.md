# Connections+

Connections+ is our own construction: each puzzle is the **union of two NYT Connections
daily puzzles**, giving 32 words that must be sorted into 8 groups of 4. Raising the
number of groups from 4 to 8 lifts the cohort base rate above the ceiling of singleton
Connections. This is the one domain we ship in full (the underlying words are public NYT
Connections content; see `source_dates`).

`manifest.json` is the single source of truth: top-level build metadata plus every puzzle
inline. (There is no separate per-puzzle directory; the manifest is the complete record.)

## Top-level fields

| Field | Meaning |
|---|---|
| `source_dataset` | `connections_100` — the pool of single NYT Connections puzzles paired from |
| `n_pairs` | 100 (one Connections+ puzzle per pair) |
| `seed` | RNG seed for the pairing/shuffle (reproducible) |
| `filters` | pairs are rejected if they share a word or a (case-insensitive) category name |
| `puzzles` | list of the 100 puzzles (schema below) |

## Per-puzzle fields (`puzzles[i]`)

| Field | Meaning |
|---|---|
| `puzzle_id` | our id, e.g. `connp_000` |
| `source_pair` | the two source puzzle ids, `[A, B]` |
| `source_dates` | the two source NYT dates, aligned to `source_pair` |
| `source_ids` | numeric ids of the two source puzzles |
| `words` | the 32 words, **shuffled** (the model sees this order) |
| `groups` | the 8 gold groups (below) |

## Group fields (`puzzles[i].groups[j]`)

| Field | Meaning |
|---|---|
| `group` | the category name |
| `members` | the 4 words in the group |
| `source` | **which source puzzle the group came from**: `"A"` = `source_pair[0]`, `"B"` = `source_pair[1]` |
| `source_puzzle_id` | the source puzzle id (redundant with `source` via `source_pair`) |
| `level` | `-1` sentinel — the original NYT colour-difficulty level (0–3) is not retained in the union |

The task is to assign each of the 32 `words` to its correct group; the 8 `groups` are the
gold answer. Model outputs for this domain are in
`../insight_4domain/all_traces.parquet` (`domain == "ConnectionsPlus"`).
