"""Tests for simulated annealing optimizer."""

import pytest
from apo.strategies.simulated_annealing import SimulatedAnnealingOptimizer
from apo.core.prompt import Prompt
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_simulated_annealing_initialization(mock_llm_provider, mock_evaluator):
    """Test simulated annealing optimizer initialization."""
    optimizer = SimulatedAnnealingOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        initial_temperature=100.0,
        cooling_rate=0.95,
        min_temperature=0.01
    )

    assert optimizer.initial_temperature == 100.0
    assert optimizer.cooling_rate == 0.95
    assert optimizer.min_temperature == 0.01


@pytest.mark.asyncio
async def test_simulated_annealing_temperature_cooling(mock_llm_provider, mock_evaluator):
    """Test temperature cooling schedule."""
    optimizer = SimulatedAnnealingOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        initial_temperature=100.0,
        cooling_rate=0.9
    )

    temp = optimizer._get_temperature(iteration=0)
    assert temp == 100.0

    temp = optimizer._get_temperature(iteration=1)
    assert abs(temp - 90.0) < 0.01

    temp = optimizer._get_temperature(iteration=2)
    assert abs(temp - 81.0) < 0.01


@pytest.mark.asyncio
async def test_simulated_annealing_acceptance_probability(mock_llm_provider, mock_evaluator):
    """Test acceptance probability calculation."""
    optimizer = SimulatedAnnealingOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator
    )

    # Better solution should always be accepted
    prob = optimizer._acceptance_probability(
        current_score=0.5,
        new_score=0.7,
        temperature=10.0
    )
    assert prob == 1.0

    # Worse solution has probability < 1
    prob = optimizer._acceptance_probability(
        current_score=0.7,
        new_score=0.5,
        temperature=10.0
    )
    assert 0.0 < prob < 1.0

    # At high temperature, worse solutions more likely
    high_temp_prob = optimizer._acceptance_probability(
        current_score=0.7,
        new_score=0.5,
        temperature=100.0
    )
    low_temp_prob = optimizer._acceptance_probability(
        current_score=0.7,
        new_score=0.5,
        temperature=1.0
    )
    assert high_temp_prob > low_temp_prob


@pytest.mark.asyncio
async def test_simulated_annealing_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete simulated annealing optimization."""
    optimizer = SimulatedAnnealingOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        initial_temperature=50.0,
        cooling_rate=0.9
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=10)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0
    assert result.iterations <= 10
    assert len(result.history) > 0


@pytest.mark.asyncio
async def test_simulated_annealing_explores_solutions(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that SA explores solution space (accepts worse solutions sometimes)."""
    optimizer = SimulatedAnnealingOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        initial_temperature=100.0,
        cooling_rate=0.95
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=15)

    # Should explore multiple solutions
    assert len(result.history) > 1
    # Best score should still be reasonable
    assert result.best_score >= 0
