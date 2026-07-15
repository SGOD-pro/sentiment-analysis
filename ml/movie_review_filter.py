"""
Filters for long-form Movies/TV reviews that are plot summaries dragging
down sentiment signal.

DO NOT apply this blind. First run the length-distribution check described
in chat. If it's not actually bimodal, this whole approach is solving a
problem you don't have and you should look elsewhere (e.g. the labels
themselves) for why Movies/TV underperforms.
"""

import re
import numpy as np
import pandas as pd

# crude sentence splitter - good enough for review text, not a full NLP
# sentence tokenizer. If you have nltk/spacy already loaded in your
# pipeline, use their sentence tokenizer instead, it's more robust to
# abbreviations ("Mr.", "e.g.") than this regex.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = _SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# OPTION A: fast baseline - keep the LAST N sentences, not the first.
# Reviewers overwhelmingly front-load plot summary and land their actual
# verdict at the end. Grabbing the first N sentences keeps the summary and
# throws away the opinion - backwards for your goal.
# ---------------------------------------------------------------------------
def extract_last_n_sentences(text: str, n: int = 3, min_sentences_to_trigger: int = 6):
    """
    Only trims reviews that are actually long (>= min_sentences_to_trigger).
    Short reviews pass through untouched - don't mangle reviews that were
    never the problem in the first place.
    """
    sentences = split_sentences(text)
    if len(sentences) < min_sentences_to_trigger:
        return text  # not long enough to be a "summary" problem, leave alone
    return ' '.join(sentences[-n:])


# ---------------------------------------------------------------------------
# OPTION B: heuristic opinion-density scoring - keep top-K sentences by
# score instead of by position. More robust than assuming opinions are
# always at the end (they usually are, but not always - some people open
# with "Loved it." then explain the plot, then close with more praise).
# ---------------------------------------------------------------------------
_OPINION_WORDS = {
    'love', 'loved', 'hate', 'hated', 'great', 'terrible', 'amazing',
    'awful', 'best', 'worst', 'boring', 'brilliant', 'disappointing',
    'disappointed', 'excellent', 'awesome', 'horrible', 'fantastic',
    'recommend', 'waste', 'enjoyed', 'enjoy', 'liked', 'disliked',
    'perfect', 'garbage', 'masterpiece', 'overrated', 'underrated',
}
_FIRST_PERSON = {'i', 'my', 'me', 'we', 'our'}


def _score_sentence(sentence: str) -> float:
    words = re.findall(r"[a-zA-Z']+", sentence.lower())
    if not words:
        return 0.0

    score = 0.0
    score += sum(1 for w in words if w in _OPINION_WORDS) * 2.0
    score += sum(1 for w in words if w in _FIRST_PERSON) * 1.0
    score += sentence.count('!') * 1.5
    # penalize very long sentences - summaries tend to run on describing plot
    if len(words) > 30:
        score -= 1.0
    return score / max(len(words), 1)  # normalize so length alone doesn't win


def extract_top_opinion_sentences(text: str, k: int = 3, min_sentences_to_trigger: int = 6):
    sentences = split_sentences(text)
    if len(sentences) < min_sentences_to_trigger:
        return text

    scored = [(s, _score_sentence(s)) for s in sentences]
    # keep top-k by score, but restore ORIGINAL ORDER so the result still
    # reads coherently instead of being scrambled
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
    top_sentences = set(s for s, _ in top)
    ordered = [s for s in sentences if s in top_sentences]
    return ' '.join(ordered)


# ---------------------------------------------------------------------------
# Apply only to the categories that actually have this problem - don't run
# this on every category, you'll mangle short, already-fine reviews for no
# reason on categories where this was never an issue.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TARGET_CATEGORIES = ["Movies_and_TV"]  # confirm this matches your actual category string

    mask = df["category"].isin(TARGET_CATEGORIES)

    # ALWAYS check before/after label distribution on the affected slice -
    # if this shifts class balance meaningfully, you need to know that,
    # not discover it three experiments later.
    print("Before, label distribution (target categories only):")
    print(df.loc[mask, "label"].value_counts(normalize=True))

    df.loc[mask, "text"] = df.loc[mask, "text"].apply(
        lambda t: extract_top_opinion_sentences(t, k=3, min_sentences_to_trigger=6)
    )

    print("\nAfter, label distribution (target categories only) - should be unchanged,")
    print("since this trims text, it doesn't drop rows:")
    print(df.loc[mask, "label"].value_counts(normalize=True))
