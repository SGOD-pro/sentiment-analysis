"""
Tests for backend ML inference service.

Verifies sentiment MLP prediction, thresholding, and issue clustering inside backend.
"""

import numpy as np
import pytest

from services.ml_inference import (
    analyze_reviews,
    apply_asymmetric_threshold,
    assign_issue_cluster,
    mlp_forward,
)


def test_mlp_forward_shape_and_probability_sum():
    """MLP forward pass should return valid probabilities summing to 1.0 for each row."""
    dummy_x = np.random.randn(5, 384).astype(np.float32)
    probs = mlp_forward(dummy_x)

    assert probs.shape == (5, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_asymmetric_threshold_behavior():
    """Positive reviews below margin threshold should fall back to neutral."""
    # Class order: 0=negative, 1=neutral, 2=positive
    # Positive with small margin:
    probs = np.array([
        [0.1, 0.44, 0.46],  # positive margin is 0.02 < 0.3 threshold -> neutral
        [0.01, 0.09, 0.90],  # positive margin is 0.81 >= 0.3 -> positive
        [0.85, 0.10, 0.05],  # negative margin is 0.75 >= 0.0 -> negative
    ], dtype=np.float32)

    preds, margins = apply_asymmetric_threshold(probs)
    assert preds[0] == 1  # fell back to neutral
    assert preds[1] == 2  # remains positive
    assert preds[2] == 0  # remains negative


def test_assign_issue_cluster():
    """Issue clustering should assign a valid cluster tag and distance."""
    dummy_emb = np.random.randn(384).astype(np.float32)
    tag, dist, source = assign_issue_cluster(dummy_emb, "Electronics")

    assert isinstance(tag, str)
    assert len(tag) > 0
    assert isinstance(dist, float)
    assert source in ("per_category", "cross_category_fallback")


def test_analyze_reviews_end_to_end():
    """analyze_reviews processes batch and attaches issue_tag to negative reviews."""
    dummy_embs = np.random.randn(3, 384).astype(np.float32)
    texts = ["Review 1", "Review 2", "Review 3"]
    categories = ["Electronics", "Books", "Home"]

    results = analyze_reviews(dummy_embs, texts, categories)

    assert len(results) == 3
    for r in results:
        assert r["sentiment"] in ("negative", "neutral", "positive")
        assert "sentiment_confidence_margin" in r
        assert "sentiment_probabilities" in r
        assert len(r["sentiment_probabilities"]) == 3
        if r["sentiment"] == "negative":
            assert r["issue_tag"] is not None
        else:
            assert r["issue_tag"] is None
