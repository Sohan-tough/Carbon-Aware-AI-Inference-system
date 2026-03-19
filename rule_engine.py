"""
rule_engine.py — Stage 1: Ultra-low-energy rule-based sentiment classifier.

This module uses handcrafted keyword lists to detect *obvious* sentiment
without loading any AI model.  When it fires it returns confidence = 1.0
so the pipeline can skip all downstream stages.

Energy cost: negligible (pure Python set operations).
"""

from __future__ import annotations

# ── Keyword lexicons ───────────────────────────────────────────────────────────
POSITIVE_KEYWORDS: set[str] = {
    "great", "excellent", "amazing", "fantastic", "wonderful", "outstanding",
    "superb", "brilliant", "awesome", "love", "loved", "perfect", "best",
    "incredible", "exceptional", "delightful", "impressive", "magnificent",
    "splendid", "terrific", "phenomenal", "remarkable", "beautiful", "happy",
    "joyful", "pleased", "satisfied", "thrilled", "excited", "glad",
}

NEGATIVE_KEYWORDS: set[str] = {
    "bad", "terrible", "worst", "horrible", "awful", "dreadful", "disgusting",
    "hate", "hated", "pathetic", "useless", "poor", "disappointing",
    "disappointed", "waste", "rubbish", "trash", "garbage", "atrocious",
    "abysmal", "disastrous", "inferior", "defective", "broken", "failed",
    "failure", "frustrated", "angry", "furious", "miserable", "unacceptable",
}

NEUTRAL_KEYWORDS: set[str] = {
    "okay", "ok", "fine", "average", "normal", "standard", "typical", "usual",
    "ordinary", "regular", "acceptable", "adequate", "sufficient", "decent",
    "fair", "moderate", "reasonable", "neutral", "balanced", "mixed", "so-so",
    "alright", "nothing", "whatever", "meh", "bland", "plain", "basic",
}


def detect_sentiment(text: str) -> dict | None:
    """
    Scan *text* for obvious positive / negative / neutral keywords.

    Returns
    -------
    dict  with keys ``label``, ``confidence``, ``stage`` if a keyword fires.
    None  if the text is ambiguous — pipeline should proceed to Stage 2.
    """
    tokens = set(text.lower().split())

    pos_hits = tokens & POSITIVE_KEYWORDS
    neg_hits = tokens & NEGATIVE_KEYWORDS
    neu_hits = tokens & NEUTRAL_KEYWORDS

    # Multiple sentiment types detected → ambiguous; let the model decide
    hit_count = sum([bool(pos_hits), bool(neg_hits), bool(neu_hits)])
    if hit_count > 1:
        return None

    if pos_hits:
        return {
            "label":      "POSITIVE",
            "confidence": 1.0,
            "stage":      "Rule Engine",
            "matched":    sorted(pos_hits),
        }

    if neg_hits:
        return {
            "label":      "NEGATIVE",
            "confidence": 1.0,
            "stage":      "Rule Engine",
            "matched":    sorted(neg_hits),
        }

    if neu_hits:
        return {
            "label":      "NEUTRAL",
            "confidence": 1.0,
            "stage":      "Rule Engine",
            "matched":    sorted(neu_hits),
        }

    return None  # No clear signal — escalate to AI
