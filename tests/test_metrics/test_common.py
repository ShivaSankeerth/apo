"""Tests for common metrics."""

import pytest
from apo.metrics.common import accuracy_score, semantic_similarity, token_efficiency


def test_accuracy_score_perfect():
    """Test accuracy with perfect predictions."""
    predictions = ["positive", "negative", "neutral"]
    targets = ["positive", "negative", "neutral"]

    score = accuracy_score(predictions, targets)
    assert score == 1.0


def test_accuracy_score_zero():
    """Test accuracy with all wrong predictions."""
    predictions = ["positive", "positive", "positive"]
    targets = ["negative", "neutral", "negative"]

    score = accuracy_score(predictions, targets)
    assert score == 0.0


def test_accuracy_score_partial():
    """Test accuracy with partial correctness."""
    predictions = ["positive", "negative", "positive"]
    targets = ["positive", "negative", "neutral"]

    score = accuracy_score(predictions, targets)
    assert abs(score - 2/3) < 0.01


def test_accuracy_score_with_whitespace():
    """Test accuracy handles whitespace correctly."""
    predictions = ["positive  ", "  negative", " neutral "]
    targets = ["positive", "negative", "neutral"]

    score = accuracy_score(predictions, targets)
    assert score == 1.0


def test_accuracy_score_empty():
    """Test accuracy with empty lists."""
    assert accuracy_score([], []) == 0.0
    assert accuracy_score(["positive"], []) == 0.0
    assert accuracy_score([], ["positive"]) == 0.0


def test_accuracy_score_mismatched_lengths():
    """Test accuracy with mismatched list lengths."""
    predictions = ["positive", "negative"]
    targets = ["positive"]

    score = accuracy_score(predictions, targets)
    assert score == 0.0


def test_semantic_similarity_identical():
    """Test semantic similarity with identical texts."""
    text1 = "This is a test sentence"
    text2 = "This is a test sentence"

    similarity = semantic_similarity(text1, text2)
    assert similarity == 1.0


def test_semantic_similarity_completely_different():
    """Test semantic similarity with completely different texts."""
    text1 = "apple banana cherry"
    text2 = "dog elephant fox"

    similarity = semantic_similarity(text1, text2)
    assert similarity == 0.0


def test_semantic_similarity_partial_overlap():
    """Test semantic similarity with partial word overlap."""
    text1 = "the quick brown fox"
    text2 = "the slow brown dog"

    similarity = semantic_similarity(text1, text2)
    assert 0.0 < similarity < 1.0
    # "the" and "brown" overlap, so similarity should be > 0


def test_semantic_similarity_case_insensitive():
    """Test that semantic similarity is case insensitive."""
    text1 = "Hello World"
    text2 = "hello world"

    similarity = semantic_similarity(text1, text2)
    assert similarity == 1.0


def test_semantic_similarity_empty_texts():
    """Test semantic similarity with empty texts."""
    assert semantic_similarity("", "") == 0.0
    assert semantic_similarity("test", "") == 0.0
    assert semantic_similarity("", "test") == 0.0


def test_token_efficiency_short_prompt():
    """Test token efficiency with short prompt."""
    prompt = "Short prompt"

    efficiency = token_efficiency(prompt, max_tokens=1000)
    assert 0.0 < efficiency <= 1.0
    # Shorter prompts should have higher efficiency


def test_token_efficiency_long_prompt():
    """Test token efficiency with long prompt."""
    prompt = " ".join(["word"] * 500)  # Very long prompt

    efficiency = token_efficiency(prompt, max_tokens=1000)
    assert 0.0 <= efficiency < 1.0
    # Longer prompts should have lower efficiency


def test_token_efficiency_comparison():
    """Test that shorter prompts have better efficiency."""
    short_prompt = "Brief"
    long_prompt = " ".join(["word"] * 100)

    short_efficiency = token_efficiency(short_prompt)
    long_efficiency = token_efficiency(long_prompt)

    assert short_efficiency > long_efficiency


def test_token_efficiency_empty_prompt():
    """Test token efficiency with empty prompt."""
    efficiency = token_efficiency("")
    assert efficiency >= 0.0


def test_token_efficiency_custom_max_tokens():
    """Test token efficiency with custom max tokens."""
    prompt = "Test prompt"

    eff1 = token_efficiency(prompt, max_tokens=100)
    eff2 = token_efficiency(prompt, max_tokens=1000)

    # Both should be valid efficiency scores
    assert 0.0 <= eff1 <= 1.0
    assert 0.0 <= eff2 <= 1.0
