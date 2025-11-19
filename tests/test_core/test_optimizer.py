"""Tests for optimizer module."""

import pytest
from unittest.mock import AsyncMock
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import EvaluationResult


class SimpleTestOptimizer(Optimizer):
    """Simple optimizer for testing base functionality."""

    async def optimize(self, initial_prompt, test_cases, max_iterations=5):
        """Simple optimization that just returns the initial prompt."""
        evaluation = await self.evaluator.evaluate(initial_prompt, test_cases)
        initial_prompt.score = evaluation.score

        return OptimizationResult(
            best_prompt=initial_prompt,
            best_score=evaluation.score,
            history=[initial_prompt],
            iterations=1,
            metadata={"test": True}
        )


@pytest.mark.asyncio
async def test_optimizer_initialization(mock_llm_provider, mock_evaluator):
    """Test optimizer initialization."""
    optimizer = SimpleTestOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator
    )

    assert optimizer.provider == mock_llm_provider
    assert optimizer.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_optimization_result_creation(sample_prompt):
    """Test optimization result creation."""
    result = OptimizationResult(
        best_prompt=sample_prompt,
        best_score=0.85,
        history=[sample_prompt],
        iterations=10,
        metadata={"converged": True}
    )

    assert result.best_prompt == sample_prompt
    assert result.best_score == 0.85
    assert len(result.history) == 1
    assert result.iterations == 10
    assert result.metadata["converged"] is True


@pytest.mark.asyncio
async def test_optimizer_convergence_check(mock_llm_provider, simple_evaluator):
    """Test convergence checking."""
    optimizer = SimpleTestOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        convergence_threshold=0.01,
        convergence_patience=3
    )

    # Test with improving scores (no convergence)
    scores = [0.5, 0.6, 0.7, 0.8]
    assert not optimizer._check_convergence(scores)

    # Test with stable scores (should converge)
    scores = [0.75, 0.751, 0.752, 0.751]
    assert optimizer._check_convergence(scores)


@pytest.mark.asyncio
async def test_optimizer_basic_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test basic optimization flow."""
    optimizer = SimpleTestOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=5)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0
    assert result.iterations >= 1
    assert len(result.history) >= 1


@pytest.mark.asyncio
async def test_optimizer_history_tracking(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that optimizer tracks history correctly."""
    optimizer = SimpleTestOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases)

    assert len(result.history) > 0
    assert all(isinstance(p, Prompt) for p in result.history)
    assert all(p.score is not None for p in result.history)


@pytest.mark.asyncio
async def test_optimizer_score_improvement(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that the best score is tracked correctly."""
    optimizer = SimpleTestOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases)

    # Best score should be the score of the best prompt
    assert result.best_score == result.best_prompt.score
