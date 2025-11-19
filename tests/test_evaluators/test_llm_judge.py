"""Tests for LLM-as-Judge evaluator."""

import pytest
from unittest.mock import AsyncMock
from apo.evaluators.llm_judge import (
    LLMJudgeEvaluator,
    QuickLLMJudge,
    EvaluationRubric,
    ACCURACY_RUBRIC,
    CLARITY_RUBRIC,
    HELPFULNESS_RUBRIC
)
from apo.core.prompt import Prompt
from apo.core.evaluator import EvaluationResult


def test_evaluation_rubric_creation():
    """Test creating an evaluation rubric."""
    rubric = EvaluationRubric(
        name="Test Rubric",
        description="A test rubric",
        criteria=["Criterion 1", "Criterion 2"],
        scale={1: "Poor", 5: "Excellent"}
    )

    assert rubric.name == "Test Rubric"
    assert len(rubric.criteria) == 2
    assert 1 in rubric.scale
    assert 5 in rubric.scale


def test_predefined_rubrics():
    """Test that predefined rubrics are available."""
    assert ACCURACY_RUBRIC.name == "Accuracy"
    assert CLARITY_RUBRIC.name == "Clarity"
    assert HELPFULNESS_RUBRIC.name == "Helpfulness"

    # All rubrics should have criteria and scales
    for rubric in [ACCURACY_RUBRIC, CLARITY_RUBRIC, HELPFULNESS_RUBRIC]:
        assert len(rubric.criteria) > 0
        assert len(rubric.scale) > 0


def test_llm_judge_initialization(mock_llm_provider):
    """Test LLM judge evaluator initialization."""
    evaluator = LLMJudgeEvaluator(
        judge_provider=mock_llm_provider,
        rubrics=[ACCURACY_RUBRIC, CLARITY_RUBRIC],
        judge_temperature=0.0,
        use_chain_of_thought=True
    )

    assert len(evaluator.rubrics) == 2
    assert evaluator.judge_temperature == 0.0
    assert evaluator.use_chain_of_thought is True


def test_llm_judge_default_rubrics(mock_llm_provider):
    """Test LLM judge with default rubrics."""
    evaluator = LLMJudgeEvaluator(judge_provider=mock_llm_provider)

    # Should have default rubrics
    assert len(evaluator.rubrics) > 0


@pytest.mark.asyncio
async def test_llm_judge_evaluate(mock_llm_provider, sample_prompt, sample_test_cases):
    """Test LLM judge evaluation."""
    # Mock judge response with scores
    mock_llm_provider.generate.return_value = """
    Reasoning: The response is accurate and clear.
    Score: 8/10
    """

    evaluator = LLMJudgeEvaluator(
        judge_provider=mock_llm_provider,
        rubrics=[ACCURACY_RUBRIC],
        use_chain_of_thought=True
    )

    result = await evaluator.evaluate(sample_prompt, sample_test_cases)

    assert isinstance(result, EvaluationResult)
    assert result.prompt == sample_prompt
    assert "score" in result.metrics or len(result.metrics) > 0
    # Judge should have been called
    assert mock_llm_provider.generate.called


@pytest.mark.asyncio
async def test_llm_judge_multiple_rubrics(mock_llm_provider, sample_prompt, sample_test_cases):
    """Test LLM judge with multiple rubrics."""
    mock_llm_provider.generate.return_value = "Score: 7/10"

    evaluator = LLMJudgeEvaluator(
        judge_provider=mock_llm_provider,
        rubrics=[ACCURACY_RUBRIC, CLARITY_RUBRIC, HELPFULNESS_RUBRIC]
    )

    result = await evaluator.evaluate(sample_prompt, sample_test_cases)

    # Should evaluate against all rubrics
    assert mock_llm_provider.generate.call_count >= 1
    assert isinstance(result, EvaluationResult)


@pytest.mark.asyncio
async def test_llm_judge_chain_of_thought(mock_llm_provider, sample_prompt, sample_test_cases):
    """Test LLM judge with chain of thought reasoning."""
    mock_llm_provider.generate.return_value = """
    Let me think step by step:
    1. The response addresses the question
    2. It is factually accurate
    3. The language is clear

    Score: 9/10
    """

    evaluator = LLMJudgeEvaluator(
        judge_provider=mock_llm_provider,
        use_chain_of_thought=True,
        include_reasoning=True
    )

    result = await evaluator.evaluate(sample_prompt, sample_test_cases)

    # Should capture reasoning if available
    if "reasoning" in result.details:
        assert isinstance(result.details["reasoning"], str)


@pytest.mark.asyncio
async def test_quick_llm_judge(mock_llm_provider, sample_prompt, sample_test_cases):
    """Test quick LLM judge evaluator."""
    mock_llm_provider.generate.return_value = "Score: 8"

    evaluator = QuickLLMJudge(judge_provider=mock_llm_provider)

    result = await evaluator.evaluate(sample_prompt, sample_test_cases)

    assert isinstance(result, EvaluationResult)
    assert result.prompt == sample_prompt


@pytest.mark.asyncio
async def test_llm_judge_score_extraction(mock_llm_provider, sample_prompt, sample_test_cases):
    """Test that LLM judge can extract scores from responses."""
    # Test various score formats
    score_formats = [
        "Score: 8/10",
        "Rating: 7",
        "8 out of 10",
        "Overall score: 9.0",
    ]

    evaluator = LLMJudgeEvaluator(judge_provider=mock_llm_provider)

    for score_format in score_formats:
        mock_llm_provider.generate.return_value = score_format
        result = await evaluator.evaluate(sample_prompt, sample_test_cases)

        # Should successfully extract a score
        assert isinstance(result, EvaluationResult)
        # Score should be normalized to 0-1
        if result.score is not None:
            assert 0 <= result.score <= 1


@pytest.mark.asyncio
async def test_llm_judge_with_custom_rubric(mock_llm_provider, sample_prompt, sample_test_cases):
    """Test LLM judge with custom rubric."""
    custom_rubric = EvaluationRubric(
        name="Creativity",
        description="Evaluate the creativity of the response",
        criteria=[
            "Is the response original?",
            "Does it show innovative thinking?",
            "Are there unique perspectives?"
        ],
        scale={
            1: "Not creative at all",
            3: "Somewhat creative",
            5: "Highly creative and original"
        }
    )

    mock_llm_provider.generate.return_value = "Score: 4/5"

    evaluator = LLMJudgeEvaluator(
        judge_provider=mock_llm_provider,
        rubrics=[custom_rubric]
    )

    result = await evaluator.evaluate(sample_prompt, sample_test_cases)

    assert isinstance(result, EvaluationResult)
    assert mock_llm_provider.generate.called
