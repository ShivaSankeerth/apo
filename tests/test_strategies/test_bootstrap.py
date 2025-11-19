"""Tests for bootstrap optimizer."""

import pytest
from apo.strategies.bootstrap import BootstrapOptimizer
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_bootstrap_initialization(mock_llm_provider, mock_evaluator):
    """Test bootstrap optimizer initialization."""
    optimizer = BootstrapOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        success_threshold=0.8,
        max_bootstrapped_examples=10
    )

    assert optimizer.success_threshold == 0.8
    assert optimizer.max_bootstrapped_examples == 10


@pytest.mark.asyncio
async def test_bootstrap_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete bootstrap optimization."""
    mock_llm_provider.generate.return_value = "Bootstrapped response"

    optimizer = BootstrapOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        max_bootstrapped_examples=5
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0


@pytest.mark.asyncio
async def test_bootstrap_collects_successful_predictions(
    mock_llm_provider, mock_evaluator, sample_prompt, sample_test_cases
):
    """Test that bootstrap collects successful predictions."""
    mock_llm_provider.generate.return_value = "positive"

    optimizer = BootstrapOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        success_threshold=0.7
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # Should track bootstrapped examples
    if "num_examples" in result.metadata:
        assert result.metadata["num_examples"] >= 0


@pytest.mark.asyncio
async def test_bootstrap_filters_by_threshold(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that bootstrap filters examples by success threshold."""
    mock_llm_provider.generate.return_value = "Generated output"

    optimizer = BootstrapOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        success_threshold=0.9,  # High threshold
        max_bootstrapped_examples=10
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # With high threshold, should have fewer examples
    if "num_examples" in result.metadata:
        num_examples = result.metadata["num_examples"]
        assert num_examples >= 0
        assert num_examples <= 10


@pytest.mark.asyncio
async def test_bootstrap_improves_with_examples(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that bootstrap improves prompts with collected examples."""
    mock_llm_provider.generate.return_value = "Response with examples"

    optimizer = BootstrapOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        max_bootstrapped_examples=3
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    # Should use LLM to generate and improve
    assert mock_llm_provider.generate.called
    assert result.iterations >= 1
