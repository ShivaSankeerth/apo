"""Hill climbing optimizer for prompt optimization."""

from typing import Any, Dict, List
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator
from apo.providers.base import LLMProvider


class HillClimbingOptimizer(Optimizer):
    """Hill climbing optimizer for prompts."""

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        num_neighbors: int = 5,
        max_iterations: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ):
        """Initialize hill climbing optimizer.

        Args:
            provider: LLM provider for generating variations
            evaluator: Evaluator for scoring prompts
            num_neighbors: Number of neighbors to generate per iteration
            max_iterations: Maximum iterations
            patience: Iterations without improvement before stopping
            verbose: Whether to print progress
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.num_neighbors = num_neighbors

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize using hill climbing.

        Args:
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and history
        """
        start_time = datetime.now()
        self._log(f"Starting hill climbing with {self.num_neighbors} neighbors per iteration")

        current_prompt = Prompt(content=initial_prompt)
        current_result = await self._evaluate_prompt(current_prompt, test_cases)

        best_prompt = current_prompt.copy()
        best_score = current_result.score
        history = [current_result]

        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            self._log(f"Iteration {iteration + 1}/{self.max_iterations} - Current score: {current_result.score:.4f}")

            # Generate neighbors
            neighbors = await self._generate_neighbors(current_prompt)

            # Evaluate neighbors
            improved = False
            for neighbor in neighbors:
                result = await self._evaluate_prompt(neighbor, test_cases)
                history.append(result)

                # Keep track of best overall
                if result.score > best_score:
                    best_score = result.score
                    best_prompt = neighbor.copy()
                    self._log(f"New best score: {best_score:.4f}")

                # Move to better neighbor
                if result.score > current_result.score:
                    current_prompt = neighbor
                    current_result = result
                    improved = True
                    no_improvement_count = 0
                    break

            if not improved:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    self._log(f"No improvement for {self.patience} iterations. Stopping.")
                    break

        end_time = datetime.now()
        self._log(f"Optimization complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            history=history,
            metadata={
                "iterations_run": iteration + 1,
                "num_neighbors": self.num_neighbors,
                "total_evaluations": len(history),
            },
            start_time=start_time,
            end_time=end_time,
        )

    async def _generate_neighbors(self, prompt: Prompt) -> List[Prompt]:
        """Generate neighbor prompts."""
        variations = await self.provider.generate_prompt_variations(
            prompt.content,
            num_variations=self.num_neighbors,
            instructions="Make small incremental improvements to the prompt."
        )

        return [Prompt(content=var) for var in variations]
