# Step 2 audit: representative judge-vs-human disagreements

Source: `data/annotations.json` (blinded human gold standard) vs `data/gold_key.json`
(deployed Gemini-3-Flash judge). Both sides scored the same traces blind to the
puzzle image, the clue, and the ground-truth answer. Scores are 0-100. The trace
language column is taken verbatim from the judge's stored key n-grams.

## The mechanism in one trace

In a single Cryptic trace (Gemini 3 Flash, puzzle cc_020) the model reasons
"enthusiastic = ARDENT, and shortening ARDENT gives ARDEN", then commits to ARDEN.
The human scores the two candidates very differently; the judge does not.

| Candidate | Human | Judge | Role in the trace |
|---|---:|---:|---|
| `ardent` | 30 | 90 | Intermediate word the model transforms, never proposed as the answer |
| `arden` (committed answer) | 99 | 95 | The actual answer |

The human marks the building block down and the answer up. The judge scores both
near the top because the language around each is locally positive. This is the
whole finding in miniature: the judge reads local linguistic positivity, the
human reads confidence that the candidate is the answer. The two measures agree
on the real answer and come apart on the building block.

## Judge over-scores building-block words

These are candidates the model manipulates as part of its reasoning but does not
put forward as the answer. The judge scores the surrounding positive language;
the human recognises the word is not being proposed.

| Domain | Candidate | Human | Judge | Trace language driving the judge's score |
|---|---|---:|---:|---|
| Rebus | `s-and` | 5 | 85 | "inside a form of s-and", "s-and" |
| Cryptic | `ardent` | 30 | 90 | "ARDENT is enthusiastic", "ARDENT almost is ARDEN", "shortening that to ARDEN" |
| Cryptic | `aus` | 15 | 60 | "Aussie gives me AUS", "AUS plus TIN", "already creates AUSTIN" |
| Rebus | `hand` | 25 | 90 | "word hand inside", "another hand", "Yes!", "this fits" |
| Rebus | `are` | 20 | 55 | "R = are", "R sounds like are" |
| VP | `D` | 0 | 95 | "eating bread is the best option", "seems correct", "solidifies my thinking" |

## The measure works on the committed answer

When the candidate really is the model's answer, judge and human agree, usually
both high. These are the pairs that feed `chosen_conf` downstream.

| Domain | Candidate (committed) | Human | Judge | Trace language |
|---|---|---:|---:|---|
| Cryptic | `egomania` | 100 | 95 | "EGOMANIA fits perfectly", "complete the word" |
| Cryptic | `erasure` | 99 | 90 | "ERASURE!", "answer is ERASURE", "ERASURE = obliteration" |
| VP | `D` | 100 | 100 | "Bingo!", "undoubtedly", "The puzzle is solved!" |

## Two honest caveats

In a minority of cases (about 10 percent) the human scores higher than the judge,
usually where the judge discounted a candidate for the model's hedging.

| Domain | Candidate | Human | Judge | Note |
|---|---|---:|---:|---|
| Rebus | `r&r` (committed) | 0 | 85 | See the worked example below. The human score is partly an answer-plausibility judgement, not a pure reading of the language. |
| Rebus | `tender` | 70 | 40 | "too literal", "bit too literal". Judge discounted the candidate for the model's hedging; the human still saw a positive lean. |

### The human standard is not a pure measure of confidence either

Puzzle 12_4 (Rebus, Claude Sonnet 4.6) is the image **"R + R ="**. The answer is
SUMMER: you *sum* the R's, so "sum R" reads as "summer". The model took the trap
reading, "R **and** R" giving R&R (rest and relaxation), and committed to it. It
was wrong (the stored result is `correct = 0`).

The human scored `r&r` 0; the judge scored it 85 off language like "settling on
R&R" and "common interpretations of that abbreviation". The gap is not cleanly
"the judge over-scored". It splits into two different things:

- The **judge** scored the language, blind to the image, and the model did sound
  fairly committed. 85 is a defensible reading of the words alone.
- The **human** marked it down mostly because R&R is an implausible answer to the
  rebus, a judgement only possible after understanding the "sum R" mechanic. That
  is answer quality leaking into a task the rubric says should score language only.

The annotator was blind to the ground truth but not to answer plausibility, and
that prior leaked in. So some of the overall judge-above-human bias is the human
marking down implausible candidates rather than the judge inflating them. The
judge, having no view of the puzzle, has no plausibility prior to leak and is in
that narrow sense more faithful to the rubric. Neither side is a clean oracle for
expressed confidence; they measure slightly different things. The committed-answer
rho of 0.82 holds despite this extra human noise, which makes it a conservative
estimate of agreement on the language itself.
