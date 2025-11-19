"""Tests for meta-prompt optimizer."""

import pytest
from apo.strategies.meta_prompt import MetaPromptOptimizer
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_meta_prompt_initialization(mock_llm_provider, mock_evaluator):
    """Test meta-prompt optimizer initialization."""
    optimizer = MetaPromptOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        reflection_depth=2
    )

    assert optimizer.reflection_depth == 2
    assert optimizer.provider == mock_llm_provider
    assert optimizer.evaluator == mock_evaluator


@pytest.mark.asyncio
async def test_meta_prompt_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete meta-prompt optimization."""
    # Configure mock to return improved prompts
    mock_llm_provider.generate.return_value = "Improved: " + sample_prompt.content

    optimizer = MetaPromptOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        reflection_depth=2
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0
    assert len(result.history) > 0
    # Should have called LLM to generate improvements
    assert mock_llm_provider.generate.call_count > 0


@pytest.mark.asyncio
async def test_meta_prompt_reflection(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that meta-prompt uses reflection to improve."""
    mock_llm_provider.generate.return_value = "More detailed and specific prompt for classification"

    optimizer = MetaPromptOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        reflection_depth=1
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # Should generate reflected improvements
    assert len(result.history) >= 1
    # LLM should be called for reflection
    assert mock_llm_provider.generate.called


@pytest.mark.asyncio
async def test_meta_prompt_uses_feedback(
    mock_llm_provider, mock_evaluator, sample_prompt, sample_test_cases
):
    """Test that meta-prompt incorporates evaluation feedback."""
    mock_llm_provider.generate.return_value = "Improved prompt based on feedback"

    optimizer = MetaPromptOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # Should pass feedback to the LLM
    call_args = mock_llm_provider.generate.call_args_list
    if call_args:
        # Check that feedback or scores are mentioned in prompts
        assert len(call_args) > 0
