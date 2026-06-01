from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize text for OCR comparison."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if s1 == s2:
        return 0

    if len(s1) == 0:
        return len(s2)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1, start=1):
        curr_row = [i]
        for j, c2 in enumerate(s2, start=1):
            insertions = prev_row[j] + 1
            deletions = curr_row[j - 1] + 1
            substitutions = prev_row[j - 1] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def character_accuracy(reference: str, hypothesis: str) -> float:
    """Compute character-level accuracy from Levenshtein distance."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if len(ref) == 0:
        return 1.0 if len(hyp) == 0 else 0.0

    dist = levenshtein_distance(ref, hyp)
    return max(0.0, 1.0 - dist / len(ref))


def word_accuracy(reference: str, hypothesis: str) -> float:
    """Compute a simple word-level accuracy."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) == 0 else 0.0

    matches = sum(
        1 for ref_word, hyp_word in zip(ref_words, hyp_words) if ref_word == hyp_word
    )
    return matches / len(ref_words)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate (WER)."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    ref_joined = "\n".join(ref_words)
    hyp_joined = "\n".join(hyp_words)

    dist = levenshtein_distance(ref_joined, hyp_joined)
    return dist / len(ref_words)