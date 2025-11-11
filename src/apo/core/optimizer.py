"""Base optimizer class and optimization result."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.providers.base import LLMProvider


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_prompt: Prompt
    best_score: float
    history: List[EvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> float:
        """Get the duration of optimization in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

    @property
    def total_evaluations(self) -> int:
        """Get the total number of evaluations performed."""
        return len(self.history)

    def get_best_n(self, n: int = 5) -> List[EvaluationResult]:
        """Get the top N results."""
        return sorted(self.history, key=lambda x: x.score, reverse=True)[:n]


class Optimizer(ABC):
    """Abstract base class for prompt optimizers."""

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        max_iterations: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ):
        """Initialize the optimizer.

        Args:
            provider: LLM provider for generating completions
            evaluator: Evaluator for scoring prompts
            max_iterations: Maximum number of optimization iterations
            patience: Number of iterations without improvement before stopping
            verbose: Whether to print progress information
        """
        self.provider = provider
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.patience = patience
        self.verbose = verbose

    @abstractmethod
    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize a prompt.

        Args:
            initial_prompt: Starting prompt to optimize
            test_cases: Test cases for evaluation
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult containing the best prompt and metadata
        """
        pass

    async def _evaluate_prompt(
        self, prompt: Prompt, test_cases: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate a single prompt.

        Args:
            prompt: Prompt to evaluate
            test_cases: Test cases for evaluation

        Returns:
            EvaluationResult
        """
        result = await self.evaluator.evaluate(prompt, test_cases)
        prompt.score = result.score
        return result

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def _check_convergence(
        self, history: List[EvaluationResult], patience: int
    ) -> bool:
        """Check if optimization has converged.

        Args:
            history: History of evaluation results
            patience: Number of iterations without improvement

        Returns:
            True if converged, False otherwise
        """
        if len(history) < patience:
            return False

        recent_scores = [result.score for result in history[-patience:]]
        best_recent = max(recent_scores)
        best_overall = max(result.score for result in history)

        return best_recent < best_overall
