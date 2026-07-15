"""
Two variants, use the right one deliberately - don't default to strict
without a reason.

text_preprocessing_strict(): collapses ALL repeated punctuation to a single
instance, including !!! and ???, and strips emojis. This throws away the
emphasis signal we specifically decided to keep in text_preprocessing.py
(see that file's _RE_EMPHASIS_PUNCT logic). Only use this if you have a
specific reason to test a maximally-stripped baseline - e.g. comparing
"does keeping emphasis punctuation actually help F1" as an ablation.

If you don't have that reason, you're just quietly undoing a decision we
already made with justification. Don't do that by accident.
"""

import re
import html
import unicodedata
import numpy as np

_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML_TAG = re.compile(r'<[^>]+>')
_RE_VIDEO_IMAGE_ID = re.compile(r'\b(?:video|image)[_-]?id[a-zA-Z0-9_-]+\b', re.IGNORECASE)
_RE_HEX_ID = re.compile(r'\b[a-fA-F0-9]{20,}\b')
_RE_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f-\x9f\u200b\u200e\u200f\ufeff]')
_RE_ALL_REPEATED_PUNCT = re.compile(r'([!?,.\-;:])\1{1,}')  # collapses EVERYTHING to 1, no exceptions
_RE_ELONGATED_LETTERS = re.compile(r'([a-zA-Z])\1{2,}')
_RE_WHITESPACE = re.compile(r'\s+')

# Emoji ranges - covers the common blocks. Not exhaustive of every unicode
# symbol block in existence, but covers what actually shows up in review text.
_RE_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols, dingbats
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "]+",
    flags=re.UNICODE,
)


def text_preprocessing_strict(text, lowercase: bool = True, min_len: int = 2):
    if not isinstance(text, str):
        return np.nan

    text = unicodedata.normalize('NFKC', text)
    text = html.unescape(text)
    text = _RE_URL.sub(' ', text)
    text = _RE_HTML_TAG.sub(' ', text)
    text = _RE_VIDEO_IMAGE_ID.sub(' ', text)
    text = _RE_HEX_ID.sub(' ', text)
    text = _RE_CONTROL_CHARS.sub(' ', text)
    text = _RE_EMOJI.sub(' ', text)

    text = _RE_ELONGATED_LETTERS.sub(r'\1\1', text)
    text = _RE_ALL_REPEATED_PUNCT.sub(r'\1', text)  # !!! -> ! , ??? -> ? , ,,,, -> ,

    text = _RE_WHITESPACE.sub(' ', text).strip()

    if lowercase:
        text = text.lower()

    if not text or len(text) < min_len:
        return np.nan

    return text


# ---------------------------------------------------------------------------
# BEFORE YOU RUN THIS ON YOUR FULL DATASET: check if it's even worth it.
# ---------------------------------------------------------------------------
def check_emoji_prevalence(df, text_col: str = "text"):
    """
    Run this FIRST. If emoji_pct is under ~1%, stripping emojis will not
    move your metrics and you're spending effort on a non-problem. Don't
    skip this and just assume it matters.
    """
    has_emoji = df[text_col].fillna("").str.contains(_RE_EMOJI, regex=True)
    pct = has_emoji.mean() * 100
    print(f"Rows containing emoji: {has_emoji.sum()} / {len(df)} ({pct:.2f}%)")
    if pct < 1.0:
        print("Under 1% - this is very likely not worth building a fix for. "
              "Confirm this number before deciding emoji handling matters.")
    return pct
