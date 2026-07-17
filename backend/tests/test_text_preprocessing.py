"""Tests for text_preprocessing."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.text_preprocessing import text_preprocessing


def test_removes_html_tags():
    assert "<b>" not in text_preprocessing("This is <b>bold</b> text")


def test_removes_urls():
    result = text_preprocessing("Check https://example.com for details")
    assert "https" not in result
    assert "example.com" not in result


def test_removes_image_video_ids():
    result = text_preprocessing("See imageidABC123_xyz for reference")
    assert "imageid" not in result.lower()


def test_removes_long_hex():
    result = text_preprocessing("Id: aabbccddeeff00112233445566 end")
    assert "aabbcc" not in result


def test_removes_uuids():
    result = text_preprocessing("Ref: 550e8400-e29b-41d4-a716-446655440000 done")
    assert "550e8400" not in result


def test_capitalizes_after_fullstop():
    result = text_preprocessing("good product. really nice. works well")
    assert "Really" in result
    assert "Works" in result


def test_capitalizes_first_char():
    result = text_preprocessing("good product")
    assert result[0] == "G"


def test_returns_none_for_empty():
    assert text_preprocessing("") is None
    assert text_preprocessing("   ") is None


def test_returns_none_for_non_string():
    assert text_preprocessing(123) is None
    assert text_preprocessing(None) is None


def test_collapses_whitespace():
    result = text_preprocessing("too   much    space")
    assert "  " not in result


def test_html_unescape():
    result = text_preprocessing("5 &gt; 3 &amp; 2 &lt; 4")
    assert ">" in result
    assert "&" in result
    assert "<" in result
