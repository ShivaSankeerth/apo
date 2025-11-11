"""Random search optimizer for prompt optimization."""

from typing import Any, Dict, List
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator
from apo.providers.base import LLMProvider


class RandomSearchOptimizer(Optimizer):
    """Random search baseline optimizer."""

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        num_samples: int = 50,
        max_iterations: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ):
        """Initialize random search optimizer.

        Args:
            provider: LLM provider for generating variations
            evaluator: Evaluator for scoring prompts
            num_samples: Number of random samples to evaluate
            max_iterations: Maximum iterations (should equal num_samples)
            patience: Not used in random search
            verbose: Whether to print progress
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.num_samples = num_samples

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize using random search.

        Args:
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and history
        """
        start_time = datetime.now()
        self._log(f"Starting random search with {self.num_samples} samples")

        # Evaluate initial prompt
        initial = Prompt(content=initial_prompt)
        initial_result = await self._evaluate_prompt(initial, test_cases)

        best_prompt = initial.copy()
        best_score = initial_result.score
        history = [initial_result]

        # Generate and evaluate random variations
        variations = await self.provider.generate_prompt_variations(
            initial_prompt,
            num_variations=self.num_samples - 1
        )

        for i, variation in enumerate(variations, 1):
            self._log(f"Evaluating sample {i}/{len(variations)}")

            prompt = Prompt(content=variation)
            result = await self._evaluate_prompt(prompt, test_cases)
            history.append(result)

            if result.score > best_score:
                best_score = result.score
                best_prompt = prompt.copy()
                self._log(f"New best score: {best_score:.4f}")

        end_time = datetime.now()
        self._log(f"Random search complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            history=history,
            metadata={
                "num_samples": self.num_samples,
                "total_evaluations": len(history),
            },
            start_time=start_time,
            end_time=end_time,
        )
