"""Few-shot example optimizer.

Optimizes not just the prompt template, but also the selection
and ordering of few-shot examples.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import random
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator
from apo.providers.base import LLMProvider


class FewShotOptimizer(Optimizer):
    """Optimizer for few-shot prompts.

    Optimizes both the template and the selection/ordering of examples.
    """

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        num_examples: int = 5,
        max_iterations: int = 30,
        patience: int = 10,
        optimize_template: bool = True,
        optimize_examples: bool = True,
        verbose: bool = True,
    ):
        """Initialize few-shot optimizer.

        Args:
            provider: LLM provider
            evaluator: Evaluator for scoring
            num_examples: Number of examples to include
            max_iterations: Maximum iterations
            patience: Early stopping patience
            optimize_template: Whether to optimize the template
            optimize_examples: Whether to optimize example selection
            verbose: Whether to print progress
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.num_examples = num_examples
        self.optimize_template = optimize_template
        self.optimize_examples = optimize_examples

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        example_pool: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize few-shot prompt.

        Args:
            initial_prompt: Template (use {examples} placeholder)
            test_cases: Test cases for evaluation
            example_pool: Pool of examples to select from
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and examples
        """
        start_time = datetime.now()
        self._log("Starting few-shot optimization")

        # Use test cases as example pool if not provided
        if example_pool is None:
            example_pool = test_cases[:20]  # Use first 20 as pool

        # Split test cases from example pool
        actual_test_cases = [tc for tc in test_cases if tc not in example_pool]
        if not actual_test_cases:
            actual_test_cases = test_cases

        # Initialize with random examples
        best_examples = random.sample(
            example_pool,
            min(self.num_examples, len(example_pool))
        )
        best_template = initial_prompt
        best_prompt = self._build_prompt(best_template, best_examples)
        best_result = await self._evaluate_prompt(best_prompt, actual_test_cases)
        best_score = best_result.score

        history = [best_result]
        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            self._log(
                f"Iteration {iteration + 1}/{self.max_iterations} - "
                f"Best score: {best_score:.4f}"
            )

            improved = False

            # Optimize example selection
            if self.optimize_examples:
                new_examples = await self._optimize_examples(
                    best_template,
                    best_examples,
                    example_pool,
                    actual_test_cases
                )

                new_prompt = self._build_prompt(best_template, new_examples)
                result = await self._evaluate_prompt(new_prompt, actual_test_cases)
                history.append(result)

                if result.score > best_score:
                    best_score = result.score
                    best_examples = new_examples
                    best_prompt = new_prompt
                    improved = True
                    self._log(f"Improved via example selection: {best_score:.4f}")

            # Optimize template
            if self.optimize_template:
                new_template = await self._optimize_template(
                    best_template,
                    best_examples,
                    actual_test_cases
                )

                new_prompt = self._build_prompt(new_template, best_examples)
                result = await self._evaluate_prompt(new_prompt, actual_test_cases)
                history.append(result)

                if result.score > best_score:
                    best_score = result.score
                    best_template = new_template
                    best_prompt = new_prompt
                    improved = True
                    self._log(f"Improved via template optimization: {best_score:.4f}")

            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check convergence
            if no_improvement_count >= self.patience:
                self._log(f"Converged after {iteration + 1} iterations")
                break

        end_time = datetime.now()
        self._log(f"Optimization complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            history=history,
            metadata={
                "iterations_run": iteration + 1,
                "total_evaluations": len(history),
                "best_examples": best_examples,
                "best_template": best_template,
            },
            start_time=start_time,
            end_time=end_time,
        )

    async def _optimize_examples(
        self,
        template: str,
        current_examples: List[Dict[str, Any]],
        example_pool: List[Dict[str, Any]],
        test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize example selection."""
        # Try different strategies
        strategies = [
            self._random_replacement,
            self._diversity_selection,
            self._coverage_selection,
        ]

        strategy = random.choice(strategies)
        return strategy(current_examples, example_pool)

    def _random_replacement(
        self,
        current_examples: List[Dict[str, Any]],
        example_pool: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Replace a random example."""
        new_examples = current_examples.copy()
        if len(example_pool) > len(current_examples):
            # Replace one random example
            idx = random.randint(0, len(new_examples) - 1)
            available = [ex for ex in example_pool if ex not in new_examples]
            if available:
                new_examples[idx] = random.choice(available)
        return new_examples

    def _diversity_selection(
        self,
        current_examples: List[Dict[str, Any]],
        example_pool: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select diverse examples."""
        # Simple diversity: shuffle and take first N
        shuffled = example_pool.copy()
        random.shuffle(shuffled)
        return shuffled[:self.num_examples]

    def _coverage_selection(
        self,
        current_examples: List[Dict[str, Any]],
        example_pool: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select examples to maximize coverage."""
        # For now, just random selection
        # Could be enhanced with actual coverage metrics
        return random.sample(example_pool, min(self.num_examples, len(example_pool)))

    async def _optimize_template(
        self,
        template: str,
        examples: List[Dict[str, Any]],
        test_cases: List[Dict[str, Any]]
    ) -> str:
        """Optimize the template given the examples."""
        examples_str = self._format_examples(examples)

        optimization_prompt = f"""You are optimizing a few-shot prompt template.

CURRENT TEMPLATE:
{template}

CURRENT EXAMPLES:
{examples_str}

The template uses {{examples}} as a placeholder for the few-shot examples.

Generate an improved version of the template that:
1. Better structures the few-shot examples
2. Provides clearer instructions
3. Helps the model learn from the examples more effectively

Return ONLY the improved template, without explanation:"""

        response = await self.provider.complete(optimization_prompt, temperature=0.7)
        return response.content.strip()

    def _build_prompt(
        self,
        template: str,
        examples: List[Dict[str, Any]]
    ) -> Prompt:
        """Build a prompt from template and examples."""
        examples_str = self._format_examples(examples)

        # Replace {examples} placeholder
        content = template.replace("{examples}", examples_str)

        return Prompt(
            content=content,
            metadata={"examples": examples, "template": template}
        )

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples for insertion into prompt."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"\nExample {i}:")
            for key, value in ex.items():
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)
