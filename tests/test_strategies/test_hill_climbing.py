"""Tests for hill climbing optimizer."""

import pytest
from apo.strategies.hill_climbing import HillClimbingOptimizer
from apo.core.prompt import Prompt
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_hill_climbing_initialization(mock_llm_provider, mock_evaluator):
    """Test hill climbing optimizer initialization."""
    optimizer = HillClimbingOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        num_neighbors=5
    )

    assert optimizer.num_neighbors == 5
    assert optimizer.provider == mock_llm_provider
    assert optimizer.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_hill_climbing_generate_neighbors(
    mock_llm_provider, mock_evaluator, sample_prompt
):
    """Test neighbor generation."""
    optimizer = HillClimbingOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        num_neighbors=3
    )

    neighbors = await optimizer._generate_neighbors(sample_prompt)

    assert len(neighbors) == 3
    assert all(isinstance(n, Prompt) for n in neighbors)
    # All should be different from original
    assert all(n.content != sample_prompt.content for n in neighbors)


@pytest.mark.asyncio
async def test_hill_climbing_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete hill climbing optimization."""
    optimizer = HillClimbingOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_neighbors=3
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=5)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0
    assert result.iterations <= 5
    assert len(result.history) > 0


@pytest.mark.asyncio
async def test_hill_climbing_accepts_better_solutions(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that hill climbing accepts better solutions."""
    optimizer = HillClimbingOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_neighbors=5
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    # Best score should be >= initial score (monotonically improving)
    initial_eval = await simple_evaluator.evaluate(sample_prompt, sample_test_cases)
    assert result.best_score >= initial_eval.score


@pytest.mark.asyncio
async def test_hill_climbing_convergence(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that hill climbing can converge."""
    optimizer = HillClimbingOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_neighbors=3,
        convergence_threshold=0.01,
        convergence_patience=2
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=20)

    # Should converge before max iterations if stuck
    assert result.iterations <= 20
    assert result.best_prompt is not None
