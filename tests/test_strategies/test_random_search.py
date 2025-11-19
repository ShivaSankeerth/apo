"""Tests for random search optimizer."""

import pytest
from apo.strategies.random_search import RandomSearchOptimizer
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_random_search_initialization(mock_llm_provider, mock_evaluator):
    """Test random search optimizer initialization."""
    optimizer = RandomSearchOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        num_samples=20
    )

    assert optimizer.num_samples == 20
    assert optimizer.provider == mock_llm_provider
    assert optimizer.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_random_search_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete random search optimization."""
    optimizer = RandomSearchOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_samples=10
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=10)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0
    assert len(result.history) > 0


@pytest.mark.asyncio
async def test_random_search_samples_multiple_prompts(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that random search samples multiple prompts."""
    optimizer = RandomSearchOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_samples=5
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=5)

    # Should try multiple different prompts
    assert len(result.history) >= 5
    # Should track the best one
    assert result.best_score == max(p.score for p in result.history if p.score is not None)


@pytest.mark.asyncio
async def test_random_search_tracks_best(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that random search tracks the best prompt found."""
    optimizer = RandomSearchOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_samples=10
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases)

    # Best prompt should have the best score
    assert result.best_prompt.score == result.best_score
    # Best score should be the maximum in history
    all_scores = [p.score for p in result.history if p.score is not None]
    assert result.best_score == max(all_scores)
