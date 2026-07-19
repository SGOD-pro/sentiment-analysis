"""
Text preprocessing for review texts before Lambda inference.

Purpose: Remove HTML, URLs, image/video IDs, hex IDs, UUIDs, control chars.
         Capitalize after full stops. Minimal — keeps punctuation and sentence structure.
Input: Raw review text string.
Output: Cleaned text string, or None if empty.
Dependencies: re, html (stdlib only)
"""

import html
import re


def text_preprocessing(text: str) -> str | None:
    """
    Minimal preprocessing for transformer embedding models.
    Keeps punctuation and sentence structure.
    Removes only obvious noise.
    """
    if not isinstance(text, str):
        return None

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove Amazon video/image ids
    text = re.sub(
        r"\b(?:video|image)id[a-zA-Z0-9_-]+\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove long hexadecimal ids
    text = re.sub(r"\b[a-fA-F0-9]{20,}\b", " ", text)

    # Remove UUIDs
    text = re.sub(
        r"\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b",
        " ",
        text,
    )

    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return None

    # Capitalize first char after full stops
    text = re.sub(r"(\.\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    # Capitalize very first character
    text = text[0].upper() + text[1:] if text else text

    return text
