"""Tests for evaluator module."""

import pytest
from apo.core.prompt import Prompt
from apo.core.evaluator import EvaluationResult, SimpleAccuracyEvaluator, MultiObjectiveEvaluator


def test_evaluation_result_score():
    """Test evaluation result score calculation."""
    prompt = Prompt(content="Test")
    result = EvaluationResult(
        prompt=prompt,
        metrics={"accuracy": 0.8, "efficiency": 0.6}
    )

    assert result.score == 0.7  # Average of 0.8 and 0.6


def test_evaluation_result_weighted_score():
    """Test weighted score calculation."""
    prompt = Prompt(content="Test")
    result = EvaluationResult(
        prompt=prompt,
        metrics={"accuracy": 0.8, "efficiency": 0.6}
    )

    weighted = result.get_weighted_score({"accuracy": 0.7, "efficiency": 0.3})
    expected = (0.8 * 0.7 + 0.6 * 0.3) / (0.7 + 0.3)
    assert abs(weighted - expected) < 0.001


@pytest.mark.asyncio
async def test_simple_accuracy_evaluator():
    """Test simple accuracy evaluator."""
    evaluator = SimpleAccuracyEvaluator()
    prompt = Prompt(content="Test prompt")

    test_cases = [
        {"input": "test1", "expected_output": "output1"},
        {"input": "test2", "expected_output": "output2"},
    ]

    result = await evaluator.evaluate(prompt, test_cases)
    assert isinstance(result, EvaluationResult)
    assert "accuracy" in result.metrics
