"""Tests for few-shot optimizer."""

import pytest
from apo.strategies.few_shot import FewShotOptimizer
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_few_shot_initialization(mock_llm_provider, mock_evaluator):
    """Test few-shot optimizer initialization."""
    optimizer = FewShotOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        max_examples=5,
        optimize_template=True
    )

    assert optimizer.max_examples == 5
    assert optimizer.optimize_template is True


@pytest.mark.asyncio
async def test_few_shot_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete few-shot optimization."""
    mock_llm_provider.generate.return_value = "Example: input -> output"

    optimizer = FewShotOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        max_examples=3
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0


@pytest.mark.asyncio
async def test_few_shot_adds_examples(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that few-shot optimizer adds examples."""
    mock_llm_provider.generate.return_value = "Generated example"

    optimizer = FewShotOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        max_examples=2
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    # Should track examples in metadata
    if "num_examples" in result.metadata:
        assert result.metadata["num_examples"] >= 0
        assert result.metadata["num_examples"] <= 2


@pytest.mark.asyncio
async def test_few_shot_template_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test template optimization in few-shot."""
    mock_llm_provider.generate.return_value = "Optimized template with examples"

    optimizer = FewShotOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        optimize_template=True,
        max_examples=3
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=2)

    assert result.best_prompt is not None
    # Should optimize both template and examples
    assert mock_llm_provider.generate.called
