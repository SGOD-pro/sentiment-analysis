import re
import html
import unicodedata
import numpy as np

# Precompiled ONCE at module load - not per row. At 200k rows this is not
# optional, it's the difference between minutes and tens of minutes.
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML_TAG = re.compile(r'<[^>]+>')
_RE_VIDEO_IMAGE_ID = re.compile(r'\b(?:video|image)[_-]?id[a-zA-Z0-9_-]+\b', re.IGNORECASE)
_RE_HEX_ID = re.compile(r'\b[a-fA-F0-9]{20,}\b')
# Added zero-width space, LTR/RTL marks, BOM - these show up in scraped
# review text and your original control-char range didn't catch them.
_RE_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f-\x9f\u200b\u200e\u200f\ufeff]')
_RE_ELONGATED = re.compile(r'([a-zA-Z])\1{2,}')  # letters only now - see below
_RE_EMPHASIS_PUNCT = re.compile(r'([!?])\1{1,}')  # !! or !!! etc -> collapse but KEEP as signal
_RE_NOISE_PUNCT = re.compile(r'([,.\-;:])\1{1,}')  # ,,,,, or ..... etc -> pure noise, collapse to 1
_RE_WHITESPACE = re.compile(r'\s+')


def text_preprocessing(text, lowercase: bool = True, min_len: int = 2):
    """
    lowercase: set to False when feeding a CASED model (e.g. distilbert-base-cased).
    Case carries real sentiment signal ("AMAZING" vs "amazing") - don't blindly
    strip it for every pipeline. Verify which checkpoint each downstream model
    actually uses before deciding this default.
    """
    if not isinstance(text, str):
        return np.nan

    # Normalize FIRST, before any regex touches the string. Collapses smart
    # quotes, full-width chars, combining accents into one consistent form -
    # otherwise visually-identical characters tokenize differently and
    # silently fragment your vocabulary.
    text = unicodedata.normalize('NFKC', text)

    text = html.unescape(text)
    text = _RE_URL.sub(' ', text)
    text = _RE_HTML_TAG.sub(' ', text)
    text = _RE_VIDEO_IMAGE_ID.sub(' ', text)
    text = _RE_HEX_ID.sub(' ', text)
    text = _RE_CONTROL_CHARS.sub(' ', text)

    # "sooooo goooood" -> "soo good". Letters only - keeps 2 chars so
    # genuine double letters ("soo") survive while runaway elongation doesn't.
    text = _RE_ELONGATED.sub(r'\1\1', text)

    # "!!!!!!!" / "?????" -> "!!" / "??". This is SIGNAL (emotional intensity),
    # not noise - don't strip it to one character, that erases the distinction
    # between "good." and "good!!!". Collapse to 2 so the model still sees
    # "this had multiple exclamation marks" without wasting tokens on the 8th one.
    text = _RE_EMPHASIS_PUNCT.sub(r'\1\1', text)

    # ",,,,," / "....." / "---" -> single char. This is NOT signal, it's almost
    # always scraping/formatting garbage. Collapse fully, unlike ! and ?.
    text = _RE_NOISE_PUNCT.sub(r'\1', text)

    text = _RE_WHITESPACE.sub(' ', text).strip()

    if lowercase:
        text = text.lower()

    if not text or len(text) < min_len:
        return np.nan

    return text
