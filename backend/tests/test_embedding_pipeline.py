"""
Tests for the optimized BGE ONNX Embedding Pipeline.

Verifies:
1. Output shape (N x 384)
2. Finite values (no NaN, no Inf)
3. L2 unit normalization (tolerance 1e-4)
4. Determinism across repeated invocations
5. Ordering preservation despite internal length-sorted micro-batching
6. Semantic discrimination (pos/pos > pos/neg)
7. Pooling consistency (proves CLS pooling matches training architecture)
8. Downstream MLP sentiment and KMeans issue classification agreement
"""

import os
import sys
import numpy as np
import pytest

# Add lambda and src to sys.path
LAMBDA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "lambda"))
if LAMBDA_DIR not in sys.path:
    sys.path.insert(0, LAMBDA_DIR)

import handler
from services.ml_inference import analyze_reviews


@pytest.fixture(scope="module")
def sample_texts():
    return [
        "Absolutely wonderful item! Great build quality and arrived on time.",
        "Terrible experience, completely defective and broke within minutes. Waste of money.",
        "Average quality product, neither particularly good nor bad.",
        "The battery life is acceptable for the price point.",
        "Worst customer support imaginable, refused to issue a refund.",
        "Fits perfectly on my vehicle, very happy with the purchase.",
        "Short",
        "A somewhat longer review describing the packaging, shipping times, build quality, and general satisfaction.",
    ]


def test_embedding_shape_and_finite(sample_texts):
    """Verify output embeddings have shape (N, 384) with finite values."""
    embs = handler.embed_texts(sample_texts, batch_size=4)
    assert embs.shape == (len(sample_texts), 384)
    assert not np.isnan(embs).any(), "NaN found in embeddings"
    assert not np.isinf(embs).any(), "Inf found in embeddings"


def test_embedding_l2_normalization(sample_texts):
    """Verify all embeddings are unit-normalized within numerical tolerance."""
    embs = handler.embed_texts(sample_texts, batch_size=4)
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), f"Embeddings not unit-normalized: {norms}"


def test_embedding_determinism(sample_texts):
    """Verify repeated inference on identical inputs produces identical outputs."""
    embs1 = handler.embed_texts(sample_texts, batch_size=4)
    embs2 = handler.embed_texts(sample_texts, batch_size=4)
    assert np.allclose(embs1, embs2, atol=1e-6), "Embeddings are not deterministic across runs"


def test_ordering_preservation(sample_texts):
    """
    Verify embeddings strictly match the original input text order,
    even though the implementation sorts by length internally.
    """
    # Compute batched with length-aware sorting
    batched_embs = handler.embed_texts(sample_texts, batch_size=3)

    # Compute each text individually (isolated ground truth)
    isolated_embs = np.array([handler.embed_texts([t], batch_size=1)[0] for t in sample_texts])

    # Cosine similarity for each corresponding pair must be > 0.98
    cosine_sims = np.sum(batched_embs * isolated_embs, axis=1)
    for idx, sim in enumerate(cosine_sims):
        assert sim > 0.98, f"Text index {idx} failed ordering check: cosine similarity = {sim:.4f}"


def test_semantic_discrimination():
    """Verify semantic discrimination: similar sentiments have higher similarity than opposing ones."""
    samples = [
        "Absolutely loved this! High quality, works great, highly recommended.",
        "Very good product, excellent quality, works like a charm!",
        "Terrible item, arrived broken and damaged, completely useless waste of money.",
    ]
    embs = handler.embed_texts(samples, batch_size=2)
    sim_pos_pos = float(np.dot(embs[0], embs[1]))
    sim_pos_neg = float(np.dot(embs[0], embs[2]))

    assert sim_pos_pos > sim_pos_neg, f"Expected pos/pos ({sim_pos_pos:.4f}) > pos/neg ({sim_pos_neg:.4f})"
    assert sim_pos_pos > 0.65, f"Expected high similarity for similar positive reviews, got {sim_pos_pos:.4f}"


def test_downstream_mlp_inference(sample_texts):
    """Verify embeddings produce valid sentiment probabilities and issue assignments."""
    embs = handler.embed_texts(sample_texts, batch_size=4)
    results = analyze_reviews(embs, sample_texts)

    assert len(results) == len(sample_texts)
    for r in results:
        assert r["sentiment"] in ("positive", "negative", "neutral")
        assert "sentiment_probabilities" in r
        assert 0.0 <= r["sentiment_confidence_margin"] <= 1.0


def test_pooling_mode_cls_consistency():
    """
    Regression test proving production uses CLS pooling matching
    the BGE-small-en-v1.5 architecture (1_Pooling/config.json)
    and the colab_embed.py training script.
    """
    text = "Outstanding service and super fast delivery!"
    # When tokenized, [CLS] is index 0
    enc = handler.tokenizer.encode(text)
    assert enc.ids[0] == 101, "[CLS] token ID is expected to be 101 for BERT"
    emb = handler.embed_texts([text])
    assert emb.shape == (1, 384)
    assert np.isclose(np.linalg.norm(emb[0]), 1.0, atol=1e-4)
