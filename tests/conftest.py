"""Shared pytest fixtures for APO tests."""

import pytest
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.core.provider import LLMProvider


@pytest.fixture
def sample_prompt() -> Prompt:
    """Sample prompt for testing."""
    return Prompt(
        content="Classify the sentiment of the following text: {text}",
        variables={"text": "I love this product!"},
        metadata={"version": "1.0"}
    )


@pytest.fixture
def sample_prompts() -> List[Prompt]:
    """List of sample prompts for testing."""
    return [
        Prompt(content="Prompt 1", score=0.7),
        Prompt(content="Prompt 2", score=0.8),
        Prompt(content="Prompt 3", score=0.6),
    ]


@pytest.fixture
def sample_test_cases() -> List[Dict[str, Any]]:
    """Sample test cases for evaluation."""
    return [
        {
            "input": {"text": "I love this product!"},
            "expected_output": "positive",
        },
        {
            "input": {"text": "This is terrible."},
            "expected_output": "negative",
        },
        {
            "input": {"text": "It's okay, nothing special."},
            "expected_output": "neutral",
        },
    ]


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    provider = AsyncMock(spec=LLMProvider)
    provider.generate.return_value = "Generated response"
    provider.model = "mock-model"
    return provider


@pytest.fixture
def mock_evaluator():
    """Mock evaluator for testing."""
    evaluator = AsyncMock(spec=Evaluator)

    async def mock_evaluate(prompt: Prompt, test_cases: List[Dict[str, Any]]):
        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": 0.75, "quality": 0.85},
            details={"predictions": ["positive", "negative", "neutral"]}
        )

    evaluator.evaluate.side_effect = mock_evaluate
    return evaluator


@pytest.fixture
def simple_evaluator():
    """Simple deterministic evaluator for testing."""
    class DeterministicEvaluator(Evaluator):
        async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
            # Simple scoring based on prompt length
            score = min(1.0, len(prompt.content) / 100)
            return EvaluationResult(
                prompt=prompt,
                metrics={"score": score},
                details={"length": len(prompt.content)}
            )

    return DeterministicEvaluator()


@pytest.fixture
def mock_responses():
    """Common mock responses for LLM providers."""
    return {
        "sentiment_positive": "positive",
        "sentiment_negative": "negative",
        "sentiment_neutral": "neutral",
        "improved_prompt": "Analyze and classify the sentiment of the given text as positive, negative, or neutral.",
        "reflection": "The prompt is clear but could be more specific about edge cases.",
    }


@pytest.fixture
def sample_rubric():
    """Sample evaluation rubric for LLM-as-Judge."""
    return {
        "name": "Accuracy",
        "description": "Evaluate the accuracy of the response",
        "criteria": [
            "Is the answer factually correct?",
            "Does it fully address the question?",
            "Are there any misleading statements?"
        ],
        "scale": {
            1: "Completely inaccurate",
            2: "Mostly inaccurate with some correct elements",
            3: "Partially accurate",
            4: "Mostly accurate with minor errors",
            5: "Completely accurate"
        }
    }
