"""
Sentence-weighted emotion scoring, replicating Ko & Geiping (2026) Eq. 15
and App. D.2: score each sentence independently with GoEmotions, then
aggregate to a message-level distribution using a character-length weighted
average. Reuses the same public classifier they cite.
"""

from __future__ import annotations

import re
import numpy as np

GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Rough groupings used for the "affiliative drift" comparison against the
# paper's Fig. 13 (agreement/positivity rising, negativity/hedging falling).
AFFILIATIVE_LABELS = ["admiration", "approval", "gratitude", "caring", "optimism"]
ADVERSARIAL_LABELS = ["disapproval", "annoyance", "anger", "disgust", "disappointment"]

_emotion_pipeline = None


def _get_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None:
        from transformers import pipeline

        _emotion_pipeline = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None,
            truncation=True,
        )
    return _emotion_pipeline


def _split_sentences(text: str) -> list[str]:
    # lightweight sentence splitter; swap for nltk/spacy if you want more care
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def score_message(text: str) -> dict[str, float]:
    """Eq. 15: character-length-weighted average of per-sentence emotion
    distributions."""
    sentences = _split_sentences(text)
    if not sentences:
        return {label: 0.0 for label in GOEMOTIONS_LABELS}

    pipe = _get_pipeline()
    outputs = pipe(sentences)  # list of list[{"label":..., "score":...}]
    weights = np.array([len(s) for s in sentences], dtype=float)
    weights = weights / weights.sum()

    agg = {label: 0.0 for label in GOEMOTIONS_LABELS}
    for w, sentence_scores in zip(weights, outputs):
        for entry in sentence_scores:
            agg[entry["label"]] += w * entry["score"]
    return agg


def affiliative_index(dist: dict[str, float]) -> float:
    return sum(dist.get(l, 0.0) for l in AFFILIATIVE_LABELS) - sum(
        dist.get(l, 0.0) for l in ADVERSARIAL_LABELS
    )
