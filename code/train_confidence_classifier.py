"""Predict whether a model's answer is correct from thinking-trace features.

Two classifiers:
  A — "Full": uses phrase-corpus features (ThinkProb scores, candidate counts,
       churn, margin) aggregated from timeline JSONs + phrase_features.csv
  B — "Lightweight": uses only surface features extractable from raw thinking
       text via regex (hedging words, answer frequency, thinking length, etc.)
       No timeline extraction needed — scales to any model with thinking output.

Evaluation: leave-one-puzzle-out CV, precision-recall curves, recall@90% precision.

Usage:
  python train_confidence_classifier.py results/thinking_20
"""

import argparse
import csv
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

GROUND_TRUTH_DIR = Path("ground_truth")


# ── Soft match (copied from existing codebase) ──────────────────────────────

def norm_answer(s):
    """Normalise answer for soft matching."""
    s = (s or "").lower().strip()
    if s.startswith("answer:"):
        s = s[7:].strip()
    # Remove articles, punctuation, collapse whitespace
    s = re.sub(r"[''`]", "'", s)
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.replace("'n'", " n ").replace("and", " n ")
    return re.sub(r"\s+", " ", s).strip()


def is_correct(answer, expected, alternatives=None):
    """Check if answer matches expected or any alternative."""
    na = norm_answer(answer)
    targets = [norm_answer(expected)]
    for alt in (alternatives or []):
        targets.append(norm_answer(alt))
    # Also try stripped version (no spaces)
    for t in list(targets):
        targets.append(t.replace(" ", ""))
    na_stripped = na.replace(" ", "")
    return any(na == t or na_stripped == t for t in targets)


# ── Load ground truth answers ────────────────────────────────────────────────

def load_ground_truths():
    """Load all ground truth files, return {puzzle_id: {answer, alternatives}}."""
    gts = {}
    for f in GROUND_TRUTH_DIR.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
            gts[d["puzzle_id"]] = {
                "answer": d["answer"],
                "alternatives": d.get("answer_alternatives", []),
            }
    return gts


# ── Feature extraction: Classifier A (full, phrase-corpus) ──────────────────

def load_phrase_features(results_dir):
    """Load phrase_features.csv into a dict keyed by (puzzle_id, model)."""
    path = results_dir / "phrase_features.csv"
    if not path.exists():
        return {}
    by_key = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["puzzle_id"], row["model"])
            by_key[key].append(row)
    return dict(by_key)


def load_timelines(results_dir):
    """Load all timeline_*.json files. Returns {puzzle_id: {model: [(pos, phrase, sim), ...]}}."""
    timelines = {}
    for f in results_dir.glob("timeline_*.json"):
        pid = f.stem.replace("timeline_", "")
        with open(f, encoding="utf-8") as fh:
            timelines[pid] = json.load(fh)
    return timelines


def extract_features_A(result, phrase_rows, timeline_entries):
    """Extract Classifier A features for one (model, puzzle) pair."""
    features = {}

    # From phrase_features.csv — find the final-answer phrase and the runner-up
    if phrase_rows:
        final_rows = [r for r in phrase_rows if int(r["is_final_answer"])]
        non_final = [r for r in phrase_rows if not int(r["is_final_answer"])]

        if final_rows:
            top = final_rows[0]
            # ThinkProb-like score: use mention_share * span_norm as proxy
            features["top1_mention_share"] = float(top["mention_share"])
            features["top1_span"] = float(top["span_norm"])
            features["top1_last_pos"] = float(top["last_pos_norm"])
            features["top1_first_pos"] = float(top["first_pos_norm"])
            features["top1_mention_count"] = int(top["mention_count"])
            features["top1_in_last_cluster"] = int(top["is_in_last_cluster"])
            features["top1_in_longest_cluster"] = int(top["is_in_longest_cluster"])
            features["top1_n_clusters"] = int(top["n_clusters_with"])

            # Margin: top1 mention_share - top2 mention_share
            if non_final:
                runner_up = max(non_final, key=lambda r: float(r["mention_share"]))
                features["margin_mention_share"] = float(top["mention_share"]) - float(runner_up["mention_share"])
                features["runner_up_mention_share"] = float(runner_up["mention_share"])
            else:
                features["margin_mention_share"] = float(top["mention_share"])
                features["runner_up_mention_share"] = 0.0
        else:
            # Final answer not in phrase corpus at all — low confidence signal
            features["top1_mention_share"] = 0.0
            features["top1_span"] = 0.0
            features["top1_last_pos"] = 1.0
            features["top1_first_pos"] = 1.0
            features["top1_mention_count"] = 0
            features["top1_in_last_cluster"] = 0
            features["top1_in_longest_cluster"] = 0
            features["top1_n_clusters"] = 0
            features["margin_mention_share"] = 0.0
            features["runner_up_mention_share"] = 0.0

        features["n_unique_phrases"] = int(phrase_rows[0]["total_unique_phrases"]) if phrase_rows else 0
        features["n_clusters_total"] = int(phrase_rows[0]["n_clusters_total"]) if phrase_rows else 0
    else:
        # No phrase data at all
        for k in ["top1_mention_share", "top1_span", "top1_last_pos", "top1_first_pos",
                   "margin_mention_share", "runner_up_mention_share"]:
            features[k] = 0.0
        for k in ["top1_mention_count", "top1_in_last_cluster", "top1_in_longest_cluster",
                   "top1_n_clusters", "n_unique_phrases", "n_clusters_total"]:
            features[k] = 0

    # Churn from timeline: count how many times the "leading phrase" changes
    if timeline_entries:
        phrases_seen = []
        for pos, phrase, sim in timeline_entries:
            phrases_seen.append(phrase)
        # Track leader changes
        if phrases_seen:
            leader_changes = 0
            current_leader = phrases_seen[0]
            freq = defaultdict(int)
            for p in phrases_seen:
                freq[p] += 1
                new_leader = max(freq, key=freq.get)
                if new_leader != current_leader:
                    leader_changes += 1
                    current_leader = new_leader
            features["churn_count"] = leader_changes
        else:
            features["churn_count"] = 0
    else:
        features["churn_count"] = 0

    # Thinking length
    features["thinking_length"] = result.get("thinking_length", 0) or 0
    features["thinking_tokens"] = result.get("thinking_tokens", 0) or 0
    features["response_time"] = result.get("response_time", 0) or 0

    # Existing scores from results.json
    features["explore_count"] = result.get("explore_count", 0) or 0
    features["revisit_count"] = result.get("revisit_count", 0) or 0
    raw_ratio = result.get("ex_rv_ratio", 0) or 0
    # ex_rv_ratio is float('inf') when revisit_count==0; cap for classifier stability
    features["ex_rv_ratio"] = min(float(raw_ratio), 100.0) if raw_ratio != 0 else 0.0

    return features


# ── Feature extraction: Classifier B (lightweight, no phrase corpus) ────────

HEDGE_PATTERNS = [
    r'\bwait\b', r'\bactually\b', r'\bhmm+\b', r'\bno[,.]', r'\bmaybe\b',
    r'\breconsider\b', r"\blet me think\b", r"\bi'm not sure\b", r'\bhold on\b',
    r"\bthat doesn't seem right\b", r'\bperhaps\b', r'\bon second thought\b',
    r"\blet me reconsider\b", r"\bthat can't be\b", r'\bwait,\b',
    # Added 2026-04-21 from ngram-discovery top negatives. Dropped `let's check`
    # and `to be sure` after they hurt LOMO — too often used by confident
    # analytical reasoners, not just hedgers.
    r'\bmight (indicate|represent|suggest)\b',
    r'\balso wonder if\b',
    r'\bother options\b',
]

CONFIDENCE_PATTERNS = [
    r'\bclearly\b', r'\bobviously\b', r'\bmust be\b', r'\bthe answer is\b',
    r'\bdefinitely\b', r"\bso it's\b", r'\bconfident\b', r'\bI think the answer\b',
    r"\bthat's it\b", r'\beureka\b', r'\byes!\b', r'\bof course\b',
    # Added 2026-04-21 from ngram-discovery top positives:
    r'\bperfect\b',
    r'\bgot it\b',
    r'\bjumps( out)?\b',
    r'\bimmediately\b',
    r'\bright!',
    r'\bright off\b',
    r'\bspot on\b',
    r'\bmakes sense\b',
]

# Explicit rejection of a specific candidate — structurally distinct from
# generic hedging. Added 2026-04-21. (task_framing_count was also proposed but
# dropped after its coefficient was ~0 — "ANSWER:" etc fire in most traces
# regardless of correctness.)
EXPLICIT_REJECTION_PATTERNS = [
    r"\bdoesn'?t fit\b",
    r"\bdoesn'?t work\b",
    r"\bdoesn'?t quite\b",
    r"\bnot quite right\b",
    r"\bcan'?t be it\b",
]

# Resolution markers for unresolved_hedge_count / unresolved_hedge_ratio.
# A hedge is "resolved" if any of these appear between it and the next hedge
# (or end of trace, up to a safety cap). Covers confidence phrases plus
# commit-oriented connectives and the explicit answer marker.
RESOLUTION_PATTERNS = CONFIDENCE_PATTERNS + [
    r'\btherefore\b',
    r'\bso,?\s',
    r'\bfinal(?:ly)?\b',
    r'\bANSWER:',
    r"\bcommit(?:ting)?\b",
    r"\bI'll go with\b",
    r"\bI'm going with\b",
]


def unresolved_hedge_stats(text, max_chars=1000):
    """Count hedges that have NO resolution marker between them and the next
    hedge (or end of trace, capped at `max_chars` after the hedge).

    Returns (unresolved_count, total_hedges).
    """
    if not text:
        return 0, 0
    # Collect hedge positions
    hedge_positions = []
    for pat in HEDGE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            hedge_positions.append(m.start())
    hedge_positions.sort()

    if not hedge_positions:
        return 0, 0

    resolution_regex = re.compile(
        "(" + "|".join(RESOLUTION_PATTERNS) + ")", re.IGNORECASE)

    n = len(hedge_positions)
    unresolved = 0
    for i, start in enumerate(hedge_positions):
        next_hedge = hedge_positions[i + 1] if i + 1 < n else len(text)
        end = min(next_hedge, start + max_chars)
        if not resolution_regex.search(text[start:end]):
            unresolved += 1
    return unresolved, n


def count_patterns(text, patterns):
    """Count total matches of a list of regex patterns in text."""
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, re.IGNORECASE))
    return total


def get_trigrams(words):
    """Extract trigrams from a word list."""
    return [tuple(words[i:i+3]) for i in range(len(words) - 2)]


def extract_features_B(result):
    """Extract Classifier B features — no phrase corpus needed."""
    thinking = result.get("thinking_content") or ""
    answer = result.get("answer") or ""
    answer_clean = norm_answer(answer)

    features = {}

    # Thinking length
    words = thinking.lower().split()
    features["thinking_word_count"] = len(words)
    features["thinking_char_count"] = len(thinking)
    features["thinking_tokens"] = result.get("thinking_tokens", 0) or 0
    features["response_time"] = result.get("response_time", 0) or 0

    # Answer frequency in thinking (soft match)
    if answer_clean and len(answer_clean) > 2:
        # Count occurrences of answer in normalised thinking
        thinking_norm = norm_answer(thinking)
        features["answer_freq"] = thinking_norm.count(answer_clean)
        # First and last position (normalised)
        first_idx = thinking_norm.find(answer_clean)
        if first_idx >= 0:
            features["answer_first_pos"] = first_idx / max(len(thinking_norm), 1)
            last_idx = thinking_norm.rfind(answer_clean)
            features["answer_last_pos"] = last_idx / max(len(thinking_norm), 1)
        else:
            features["answer_first_pos"] = 1.0  # not found = maximally late
            features["answer_last_pos"] = 1.0
    else:
        features["answer_freq"] = 0
        features["answer_first_pos"] = 1.0
        features["answer_last_pos"] = 1.0

    # Hedging vs confidence words
    features["hedge_count"] = count_patterns(thinking, HEDGE_PATTERNS)
    features["confidence_count"] = count_patterns(thinking, CONFIDENCE_PATTERNS)
    total_signals = features["hedge_count"] + features["confidence_count"] + 1
    features["hedge_ratio"] = features["hedge_count"] / total_signals

    # Explicit rejection of a specific candidate. Added 2026-04-21.
    features["explicit_rejection_count"] = count_patterns(thinking, EXPLICIT_REJECTION_PATTERNS)

    # Unresolved-hedge features — ACTIVE. Zero classifier-level AUC-PR lift
    # (the features are highly correlated with hedge_count by construction),
    # but cascade validation 2026-04-21 showed they recover the unique 89%
    # cell on Flash→GPT5→Gem3Pro text-only signal, and open cheaper 2-stage
    # operating points at 87%. Net: keep active.
    unres, total_hedges = unresolved_hedge_stats(thinking)
    features["unresolved_hedge_count"] = unres
    features["unresolved_hedge_ratio"] = (unres / total_hedges) if total_hedges > 0 else 0.0

    # Revision markers
    features["revision_count"] = len(re.findall(
        r'\b(no,|wait,|actually,|but |however )', thinking, re.IGNORECASE))

    # Punctuation signals
    features["question_marks"] = thinking.count("?")
    features["exclamation_marks"] = thinking.count("!")

    # Paragraph / section count (proxy for structured thinking)
    features["paragraph_count"] = max(1, len(re.findall(r'\n\s*\n', thinking)))

    # Answer length (short answers tend to be more confident)
    features["answer_word_count"] = len(answer_clean.split()) if answer_clean else 0

    # ── N-gram repetition/diversity features (1-5) ───────────────────────────

    trigrams = get_trigrams(words)
    n_trigrams = max(len(trigrams), 1)

    # 1. Bigram repetition rate
    bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
    n_bigrams = max(len(bigrams), 1)
    from collections import Counter
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
    features["bigram_repetition_rate"] = repeated_bigrams / n_bigrams

    # Trigram repetition rate
    trigram_counts = Counter(trigrams)
    repeated_trigrams = sum(c - 1 for c in trigram_counts.values() if c > 1)
    features["trigram_repetition_rate"] = repeated_trigrams / n_trigrams

    # 2. Unique trigram ratio
    features["unique_trigram_ratio"] = len(set(trigrams)) / n_trigrams

    # 3. Trigram hapax rate (fraction appearing exactly once)
    hapax = sum(1 for c in trigram_counts.values() if c == 1)
    features["trigram_hapax_rate"] = hapax / max(len(trigram_counts), 1)

    # 4. Top-5 trigram concentration
    top5 = sum(c for _, c in trigram_counts.most_common(5))
    features["top5_trigram_concentration"] = top5 / n_trigrams

    # 5. Answer n-gram dominance — overlap of answer words with top-10 trigrams
    answer_words = set(answer_clean.split()) if answer_clean else set()
    if answer_words and trigram_counts:
        top10_trigram_words = set()
        for tri, _ in trigram_counts.most_common(10):
            top10_trigram_words.update(tri)
        features["answer_ngram_dominance"] = (
            len(answer_words & top10_trigram_words) / max(len(answer_words), 1))
    else:
        features["answer_ngram_dominance"] = 0.0

    # ── Structural/sequential features (6-10) ───────────────────────────────

    # 6. Vocabulary growth rate — ratio of unique words in last third vs first third
    n_words = len(words)
    if n_words >= 6:
        third = n_words // 3
        first_third_vocab = len(set(words[:third]))
        last_third_vocab = len(set(words[-third:]))
        # New words introduced in last third relative to first third
        features["vocab_growth_ratio"] = last_third_vocab / max(first_third_vocab, 1)
        # Also: fraction of last-third words that are new (never seen before)
        seen_before = set(words[:-third])
        last_words = words[-third:]
        new_in_last = sum(1 for w in last_words if w not in seen_before)
        features["late_novelty_rate"] = new_in_last / max(len(last_words), 1)
    else:
        features["vocab_growth_ratio"] = 1.0
        features["late_novelty_rate"] = 0.0

    # 7. Hedge ratio shift — hedging in last third vs first third
    if n_words >= 6:
        third_chars = len(thinking) // 3
        first_text = thinking[:third_chars]
        last_text = thinking[-third_chars:]
        hedge_first = count_patterns(first_text, HEDGE_PATTERNS) + 1
        hedge_last = count_patterns(last_text, HEDGE_PATTERNS) + 1
        features["hedge_shift"] = hedge_last / hedge_first  # <1 = converging
    else:
        features["hedge_shift"] = 1.0

    # 8. Late answer introduction — did the answer first appear in last 20%?
    if answer_clean and len(answer_clean) > 2:
        thinking_lower = thinking.lower()
        first_idx = thinking_lower.find(answer_clean)
        if first_idx >= 0:
            features["answer_in_last_20pct"] = int(
                first_idx > 0.8 * len(thinking_lower))
            features["answer_intro_pos"] = first_idx / max(len(thinking_lower), 1)
        else:
            features["answer_in_last_20pct"] = 1  # never found = treat as late
            features["answer_intro_pos"] = 1.0
    else:
        features["answer_in_last_20pct"] = 1
        features["answer_intro_pos"] = 1.0

    # 9. Formatted answer — answer appears in quotes, bold, or caps in thinking
    if answer_clean and len(answer_clean) > 2:
        answer_upper = answer_clean.upper()
        # Check for quoted answer
        quoted = bool(re.search(
            r'["\u201c]' + re.escape(answer_clean) + r'["\u201d]',
            thinking.lower()))
        # Check for CAPS answer (at least 3 chars)
        caps_present = answer_upper in thinking.upper() and len(answer_clean) >= 3
        # Check for bold (**answer**)
        bold = bool(re.search(
            r'\*\*' + re.escape(answer_clean) + r'\*\*',
            thinking.lower()))
        features["answer_formatted"] = int(quoted or bold)
        features["answer_in_caps"] = int(caps_present and
            answer_upper in thinking)  # actually uppercase in source
    else:
        features["answer_formatted"] = 0
        features["answer_in_caps"] = 0

    # 10. Self-correction density
    correction_patterns = [
        r"\bno, it's\b", r"\bthat's wrong\b", r"\bnot .{1,20} but\b",
        r"\bi was wrong\b", r"\bcorrection\b", r"\blet me correct\b",
        r"\bthat's not right\b", r"\bactually, it should be\b",
        r"\bwait, no\b", r"\bsorry,\b",
    ]
    features["self_correction_count"] = count_patterns(thinking, correction_patterns)

    # ── Self-reported confidence (if the confidence-prompt pass was run) ───
    # Included as a feature so LR learns the optimal blend with other signals.
    # Missing values default to 0.5 (neutral). Present on ~98% of rows in practice.
    raw_conf = result.get("conf_confidence")
    if raw_conf is not None:
        features["self_reported_conf"] = max(0.0, min(1.0, float(raw_conf) / 100.0))
        features["self_reported_present"] = 1
    else:
        features["self_reported_conf"] = 0.5
        features["self_reported_present"] = 0

    # ── Tier 1 inspire/revisit proxies (regex only, no phrase pipeline) ─────

    # Hedge density in last third — late revisit = low confidence
    if n_words >= 6 and thinking:
        tlen = len(thinking)
        last_third_text = thinking[-tlen // 3:]
        last_third_words = max(len(last_third_text.split()), 1)
        features["hedge_density_last_third"] = (
            count_patterns(last_third_text, HEDGE_PATTERNS) / last_third_words)
    else:
        features["hedge_density_last_third"] = 0.0

    # Hedge position variance — are hedges spread out (churn) or clustered?
    # Low variance = hedges at one spot (one reconsideration). High = many cycles.
    hedge_positions = []
    for pat in HEDGE_PATTERNS:
        for m in re.finditer(pat, thinking, re.IGNORECASE):
            hedge_positions.append(m.start() / max(len(thinking), 1))
    if len(hedge_positions) >= 2:
        mean_p = sum(hedge_positions) / len(hedge_positions)
        var_p = sum((p - mean_p) ** 2 for p in hedge_positions) / len(hedge_positions)
        features["hedge_position_variance"] = var_p
    else:
        features["hedge_position_variance"] = 0.0

    # Candidate-switch count — "wait/actually/no" followed by a capitalized word
    # or quoted phrase within the next ~30 chars. Surface proxy for churn.
    switch_pat = re.compile(
        r'\b(?:wait|actually|no|hmm|hold on)\b[^.?!\n]{0,30}?'
        r'(?:["\u201c\u2018][A-Za-z0-9][^"\u201d\u2019]{1,30}["\u201d\u2019]'
        r'|\b[A-Z][A-Za-z]{2,}\b)',
        re.IGNORECASE)
    features["candidate_switch_count"] = len(switch_pat.findall(thinking))

    # Commit distance — chars between last hedge and last answer mention.
    # Large = clean commit after reconsidering; small/negative = hedged to the end.
    if hedge_positions and answer_clean and len(answer_clean) > 2:
        thinking_lower = thinking.lower()
        last_answer_idx = thinking_lower.rfind(answer_clean)
        if last_answer_idx >= 0:
            last_hedge_frac = max(hedge_positions)
            last_hedge_idx = int(last_hedge_frac * len(thinking))
            # Normalise to thinking length so features are comparable across lengths
            commit_chars = last_answer_idx - last_hedge_idx
            features["commit_distance"] = commit_chars / max(len(thinking), 1)
        else:
            features["commit_distance"] = 0.0
    else:
        features["commit_distance"] = 0.0

    return features


# ── Build dataset ────────────────────────────────────────────────────────────

def build_dataset(results_dir):
    """Build feature matrices for both classifiers."""
    results_path = results_dir / "results.json"
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    gts = load_ground_truths()
    phrase_data = load_phrase_features(results_dir)
    timelines = load_timelines(results_dir)

    rows_A = []
    rows_B = []
    labels = []
    meta = []  # (model, puzzle_id) for CV splits

    for r in results:
        if not r.get("thinking_content"):
            continue
        pid = r["puzzle_id"]
        model = r["display"]
        if pid not in gts:
            continue

        # Label
        gt = gts[pid]
        correct = is_correct(r.get("answer", ""), gt["answer"], gt["alternatives"])

        # Classifier A features
        phrase_key = (pid, model)
        phrase_rows = phrase_data.get(phrase_key, [])
        tl = timelines.get(pid, {}).get(model, [])
        feat_A = extract_features_A(r, phrase_rows, tl)

        # Classifier B features (surface text)
        feat_B = extract_features_B(r)

        # B-augmented: add the structural A features B doesn't already capture
        # (Tier 2 — inspire/revisit signals not derivable from raw text).
        # Pruned: ex_rv_ratio and top1_in_longest_cluster had near-zero coefficients
        # in the initial 6-feature addition (|coef| < 0.05) — dropped.
        for k in ("churn_count", "top1_in_last_cluster",
                  "explore_count", "revisit_count"):
            if k in feat_A:
                feat_B[k] = feat_A[k]

        rows_A.append(feat_A)
        rows_B.append(feat_B)
        labels.append(int(correct))
        meta.append({"model": model, "puzzle_id": pid})

    return rows_A, rows_B, labels, meta


def dicts_to_matrix(rows):
    """Convert list of feature dicts to numpy matrix + feature names."""
    if not rows:
        return np.array([]), []
    keys = sorted(rows[0].keys())
    X = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    return X, keys


# ── Evaluation ───────────────────────────────────────────────────────────────

def leave_one_puzzle_out_cv(X, y, meta, model_cls):
    """LOPO cross-validation. Returns (true_labels, predicted_probabilities)."""
    from sklearn.preprocessing import StandardScaler

    puzzles = sorted(set(m["puzzle_id"] for m in meta))
    y = np.array(y)
    all_true = []
    all_prob = []

    for held_out in puzzles:
        train_idx = [i for i, m in enumerate(meta) if m["puzzle_id"] != held_out]
        test_idx = [i for i, m in enumerate(meta) if m["puzzle_id"] == held_out]
        if not train_idx or not test_idx:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = model_cls()
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        all_true.extend(y_test.tolist())
        all_prob.extend(probs.tolist())

    return np.array(all_true), np.array(all_prob)


def leave_one_model_out_cv(X, y, meta, model_cls):
    """LOMO cross-validation. Returns (true_labels, predicted_probabilities)."""
    from sklearn.preprocessing import StandardScaler

    models = sorted(set(m["model"] for m in meta))
    y = np.array(y)
    all_true = []
    all_prob = []

    for held_out in models:
        train_idx = [i for i, m in enumerate(meta) if m["model"] != held_out]
        test_idx = [i for i, m in enumerate(meta) if m["model"] == held_out]
        if not train_idx or not test_idx:
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = model_cls()
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:, 1]
        all_true.extend(y_test.tolist())
        all_prob.extend(probs.tolist())

    return np.array(all_true), np.array(all_prob)


def evaluate_predictions(y_true, y_prob, label=""):
    """Print precision-recall stats and return key metrics."""
    from sklearn.metrics import (
        precision_recall_curve, average_precision_score,
        roc_auc_score, brier_score_loss,
    )

    ap = average_precision_score(y_true, y_prob)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = float('nan')
    brier = brier_score_loss(y_true, y_prob)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Find recall at ~90% precision
    recall_at_90 = 0.0
    threshold_at_90 = 1.0
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= 0.90 and r > recall_at_90:
            recall_at_90 = r
            threshold_at_90 = t

    # Find recall at ~80% precision
    recall_at_80 = 0.0
    threshold_at_80 = 1.0
    for p, r, t in zip(precisions, recalls, thresholds):
        if p >= 0.80 and r > recall_at_80:
            recall_at_80 = r
            threshold_at_80 = t

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Samples: {len(y_true)}  (correct: {int(y_true.sum())}, "
          f"wrong: {int(len(y_true) - y_true.sum())})")
    print(f"  Base rate (always predict correct): {y_true.mean():.1%}")
    print(f"  AUC-PR:  {ap:.3f}")
    print(f"  AUC-ROC: {auc_roc:.3f}")
    print(f"  Brier:   {brier:.3f}")
    print(f"  Recall @ 90% precision: {recall_at_90:.1%}  (threshold={threshold_at_90:.3f})")
    print(f"  Recall @ 80% precision: {recall_at_80:.1%}  (threshold={threshold_at_80:.3f})")
    print()

    # Threshold table
    print(f"  {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'Accept':>10s}  {'Correct':>10s}")
    for t_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predicted = y_prob >= t_val
        if predicted.sum() == 0:
            continue
        prec = y_true[predicted].mean()
        rec = y_true[predicted].sum() / max(y_true.sum(), 1)
        print(f"  {t_val:>10.1f}  {prec:>10.1%}  {rec:>10.1%}  "
              f"{predicted.sum():>10d}  {int(y_true[predicted].sum()):>10d}")

    return {"ap": ap, "auc_roc": auc_roc, "brier": brier,
            "recall_at_90": recall_at_90, "recall_at_80": recall_at_80}


def print_feature_importance(X, y, feature_names, model_cls, label=""):
    """Train on full data and print feature importances."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = model_cls()
    clf.fit(X_scaled, y)

    if hasattr(clf, 'coef_'):
        importances = clf.coef_[0]
    elif hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    else:
        return

    pairs = sorted(zip(feature_names, importances), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Feature importances ({label}):")
    for name, imp in pairs:
        direction = "+" if imp > 0 else "-"
        print(f"    {direction} {name:30s}  {imp:+.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Path to results directory")
    parser.add_argument("--exclude-models", default="",
                        help="Comma-separated model display names to drop before CV")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    print("Building dataset...")
    rows_A, rows_B, labels, meta = build_dataset(results_dir)

    excl = set(m.strip() for m in args.exclude_models.split(",") if m.strip())
    if excl:
        print(f"  Excluding models: {sorted(excl)}")
        keep = [i for i, m in enumerate(meta) if m["model"] not in excl]
        rows_A = [rows_A[i] for i in keep]
        rows_B = [rows_B[i] for i in keep]
        labels = [labels[i] for i in keep]
        meta = [meta[i] for i in keep]

    print(f"  {len(labels)} samples, {sum(labels)} correct, {len(labels) - sum(labels)} wrong")
    print(f"  Base rate: {sum(labels)/len(labels):.1%}")
    print(f"  Models: {sorted(set(m['model'] for m in meta))}")
    print(f"  Puzzles: {len(set(m['puzzle_id'] for m in meta))}")

    X_A, names_A = dicts_to_matrix(rows_A)
    X_B, names_B = dicts_to_matrix(rows_B)
    y = np.array(labels)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    lr = lambda: LogisticRegression(max_iter=1000, C=1.0)
    rf = lambda: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

    # ── Classifier A ──
    print("\n" + "#" * 60)
    print("  CLASSIFIER A — Full (phrase-corpus features)")
    print("#" * 60)

    for name, model_cls in [("LogisticRegression", lr), ("RandomForest", rf)]:
        y_true, y_prob = leave_one_puzzle_out_cv(X_A, y, meta, model_cls)
        evaluate_predictions(y_true, y_prob, f"A / {name} / Leave-one-puzzle-out")

        y_true, y_prob = leave_one_model_out_cv(X_A, y, meta, model_cls)
        evaluate_predictions(y_true, y_prob, f"A / {name} / Leave-one-model-out")

    print_feature_importance(X_A, y, names_A, lr, "A / LogisticRegression")

    # ── Classifier B ──
    print("\n" + "#" * 60)
    print("  CLASSIFIER B — Lightweight (no phrase corpus)")
    print("#" * 60)

    for name, model_cls in [("LogisticRegression", lr), ("RandomForest", rf)]:
        y_true, y_prob = leave_one_puzzle_out_cv(X_B, y, meta, model_cls)
        evaluate_predictions(y_true, y_prob, f"B / {name} / Leave-one-puzzle-out")

        y_true, y_prob = leave_one_model_out_cv(X_B, y, meta, model_cls)
        evaluate_predictions(y_true, y_prob, f"B / {name} / Leave-one-model-out")

    print_feature_importance(X_B, y, names_B, lr, "B / LogisticRegression")

    # ── Comparison ──
    print("\n" + "#" * 60)
    print("  COMPARISON SUMMARY")
    print("#" * 60)
    print("\n  Running final comparison (LR, LOPO)...")

    y_true_A, y_prob_A = leave_one_puzzle_out_cv(X_A, y, meta, lr)
    y_true_B, y_prob_B = leave_one_puzzle_out_cv(X_B, y, meta, lr)

    from sklearn.metrics import average_precision_score
    ap_A = average_precision_score(y_true_A, y_prob_A)
    ap_B = average_precision_score(y_true_B, y_prob_B)
    print(f"\n  Classifier A (full)       AUC-PR = {ap_A:.3f}")
    print(f"  Classifier B (lightweight) AUC-PR = {ap_B:.3f}")
    print(f"  Base rate                          = {y_true_A.mean():.3f}")
    if ap_B >= ap_A * 0.90:
        print("\n  >> B is within 90% of A — lightweight features are competitive.")
    else:
        print(f"\n  >> A outperforms B by {(ap_A - ap_B)/ap_A:.0%} — phrase corpus adds value.")


if __name__ == "__main__":
    main()
