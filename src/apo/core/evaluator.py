"""Evaluation system for prompts."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from apo.core.prompt import Prompt


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt."""

    prompt: Prompt
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Get the overall score (average of all metrics)."""
        if not self.metrics:
            return 0.0
        return sum(self.metrics.values()) / len(self.metrics)

    def get_weighted_score(self, weights: Dict[str, float]) -> float:
        """Get a weighted score based on metric weights."""
        total_score = 0.0
        total_weight = 0.0

        for metric, value in self.metrics.items():
            weight = weights.get(metric, 1.0)
            total_score += value * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0


class Evaluator(ABC):
    """Abstract base class for prompt evaluators."""

    def __init__(self, metric_weights: Optional[Dict[str, float]] = None):
        """Initialize the evaluator.

        Args:
            metric_weights: Weights for different metrics in multi-objective optimization
        """
        self.metric_weights = metric_weights or {}

    @abstractmethod
    async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate a prompt against test cases.

        Args:
            prompt: The prompt to evaluate
            test_cases: List of test cases to evaluate against

        Returns:
            EvaluationResult containing metrics and metadata
        """
        pass

    async def evaluate_batch(
        self, prompts: List[Prompt], test_cases: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """Evaluate multiple prompts.

        Args:
            prompts: List of prompts to evaluate
            test_cases: List of test cases to evaluate against

        Returns:
            List of EvaluationResults
        """
        results = []
        for prompt in prompts:
            result = await self.evaluate(prompt, test_cases)
            results.append(result)
        return results


class SimpleAccuracyEvaluator(Evaluator):
    """Simple evaluator that checks for exact matches."""

    async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate prompt based on exact match accuracy."""
        if not test_cases:
            return EvaluationResult(prompt=prompt, metrics={"accuracy": 0.0})

        correct = 0
        for test_case in test_cases:
            # This is a simplified example - in practice, you'd call an LLM here
            # and compare the output with expected_output
            if "expected_output" in test_case:
                # Placeholder logic
                correct += 1

        accuracy = correct / len(test_cases)
        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": accuracy},
            metadata={"total_cases": len(test_cases), "correct": correct}
        )


class MultiObjectiveEvaluator(Evaluator):
    """Evaluator that combines multiple evaluation criteria."""

    def __init__(
        self,
        evaluators: List[Evaluator],
        metric_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize with multiple evaluators.

        Args:
            evaluators: List of evaluators to combine
            metric_weights: Weights for different metrics
        """
        super().__init__(metric_weights)
        self.evaluators = evaluators

    async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate using all evaluators and combine results."""
        all_metrics: Dict[str, float] = {}
        all_metadata: Dict[str, Any] = {}

        for evaluator in self.evaluators:
            result = await evaluator.evaluate(prompt, test_cases)
            all_metrics.update(result.metrics)
            all_metadata[evaluator.__class__.__name__] = result.metadata

        return EvaluationResult(
            prompt=prompt,
            metrics=all_metrics,
            metadata=all_metadata
        )
