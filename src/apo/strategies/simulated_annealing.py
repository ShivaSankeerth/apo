"""Simulated annealing optimizer for prompt optimization."""

import math
import random
from typing import Any, Dict, List
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator
from apo.providers.base import LLMProvider


class SimulatedAnnealingOptimizer(Optimizer):
    """Simulated annealing optimizer for prompts."""

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 100,
        patience: int = 20,
        verbose: bool = True,
    ):
        """Initialize simulated annealing optimizer.

        Args:
            provider: LLM provider for generating variations
            evaluator: Evaluator for scoring prompts
            initial_temperature: Starting temperature
            cooling_rate: Rate at which temperature decreases
            min_temperature: Minimum temperature before stopping
            max_iterations: Maximum iterations
            patience: Iterations without improvement before stopping
            verbose: Whether to print progress
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize using simulated annealing.

        Args:
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and history
        """
        start_time = datetime.now()
        self._log(f"Starting simulated annealing with T={self.initial_temperature}, cooling={self.cooling_rate}")

        current_prompt = Prompt(content=initial_prompt)
        current_result = await self._evaluate_prompt(current_prompt, test_cases)

        best_prompt = current_prompt.copy()
        best_score = current_result.score
        history = [current_result]

        temperature = self.initial_temperature
        no_improvement_count = 0
        iteration = 0

        while temperature > self.min_temperature and iteration < self.max_iterations:
            iteration += 1
            self._log(
                f"Iteration {iteration} - T={temperature:.4f}, "
                f"Current={current_result.score:.4f}, Best={best_score:.4f}"
            )

            # Generate neighbor
            neighbor = await self._generate_neighbor(current_prompt, temperature)
            neighbor_result = await self._evaluate_prompt(neighbor, test_cases)
            history.append(neighbor_result)

            # Update best
            if neighbor_result.score > best_score:
                best_score = neighbor_result.score
                best_prompt = neighbor.copy()
                no_improvement_count = 0
                self._log(f"New best score: {best_score:.4f}")
            else:
                no_improvement_count += 1

            # Acceptance probability
            delta = neighbor_result.score - current_result.score
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_prompt = neighbor
                current_result = neighbor_result
                if delta > 0:
                    self._log(f"Accepted better neighbor (Δ={delta:.4f})")
                else:
                    self._log(f"Accepted worse neighbor (Δ={delta:.4f}, p={math.exp(delta/temperature):.4f})")

            # Cool down
            temperature *= self.cooling_rate

            # Check convergence
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
                "iterations_run": iteration,
                "final_temperature": temperature,
                "initial_temperature": self.initial_temperature,
                "cooling_rate": self.cooling_rate,
                "total_evaluations": len(history),
            },
            start_time=start_time,
            end_time=end_time,
        )

    async def _generate_neighbor(self, prompt: Prompt, temperature: float) -> Prompt:
        """Generate a neighbor prompt based on current temperature."""
        # Higher temperature -> more variation
        variation_strength = "significant" if temperature > 0.5 else "small"

        neighbor_prompt = f"""Make a {variation_strength} variation to this prompt:

{prompt.content}

Return only the modified prompt without explanation:"""

        response = await self.provider.complete(
            neighbor_prompt,
            temperature=min(0.9, temperature + 0.3)
        )
        return Prompt(content=response.content.strip())
