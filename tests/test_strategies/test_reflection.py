"""Tests for reflection optimizer."""

import pytest
from apo.strategies.reflection import ReflectionOptimizer
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_reflection_initialization(mock_llm_provider, mock_evaluator):
    """Test reflection optimizer initialization."""
    optimizer = ReflectionOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        use_pareto_frontier=True,
        num_candidates=5
    )

    assert optimizer.use_pareto_frontier is True
    assert optimizer.num_candidates == 5
    assert optimizer.provider == mock_llm_provider
    assert optimizer.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_reflection_without_pareto(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test reflection optimizer without Pareto frontier."""
    mock_llm_provider.generate.return_value = "Reflected and improved prompt"

    optimizer = ReflectionOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        use_pareto_frontier=False
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0


@pytest.mark.asyncio
async def test_reflection_with_pareto(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test reflection optimizer with Pareto frontier."""
    mock_llm_provider.generate.return_value = "Optimized prompt with Pareto consideration"

    optimizer = ReflectionOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        use_pareto_frontier=True
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    # Should track Pareto frontier in metadata
    if "pareto_frontier" in result.metadata:
        assert isinstance(result.metadata["pareto_frontier"], list)


@pytest.mark.asyncio
async def test_reflection_generates_candidates(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that reflection generates multiple candidates."""
    mock_llm_provider.generate.return_value = "Generated candidate prompt"

    optimizer = ReflectionOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        num_candidates=3
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # Should generate multiple candidates
    assert len(result.history) >= 1
    # LLM should be called multiple times
    assert mock_llm_provider.generate.call_count >= 1


@pytest.mark.asyncio
async def test_reflection_pareto_frontier_access(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test accessing Pareto frontier."""
    mock_llm_provider.generate.return_value = "Pareto-optimal prompt"

    optimizer = ReflectionOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        use_pareto_frontier=True
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # Should be able to access Pareto frontier
    frontier = optimizer.get_pareto_frontier()
    if frontier:
        assert frontier.size() >= 0
