"""Bootstrap optimizer (DSPy-inspired).

Automatically generates few-shot examples from successful predictions.
"""

from typing import Any, Dict, List
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator
from apo.providers.base import LLMProvider


class BootstrapOptimizer(Optimizer):
    """Bootstrap optimizer that generates examples from successful runs.

    Inspired by DSPy's BootstrapFewShot optimizer.
    """

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        max_bootstrapped_examples: int = 10,
        max_iterations: int = 5,
        patience: int = 3,
        verbose: bool = True,
        success_threshold: float = 0.8,
    ):
        """Initialize bootstrap optimizer.

        Args:
            provider: LLM provider
            evaluator: Evaluator for scoring
            max_bootstrapped_examples: Maximum examples to bootstrap
            max_iterations: Maximum iterations
            patience: Early stopping patience
            verbose: Whether to print progress
            success_threshold: Threshold for considering a prediction successful
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.max_bootstrapped_examples = max_bootstrapped_examples
        self.success_threshold = success_threshold

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize by bootstrapping examples.

        Args:
            initial_prompt: Template with {examples} placeholder
            test_cases: Test cases for evaluation and bootstrapping
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with bootstrapped prompt
        """
        start_time = datetime.now()
        self._log("Starting bootstrap optimization")

        # Split test cases into bootstrap and validation sets
        split_idx = int(len(test_cases) * 0.7)
        bootstrap_cases = test_cases[:split_idx]
        validation_cases = test_cases[split_idx:]

        if not validation_cases:
            validation_cases = test_cases

        # Start with zero-shot prompt
        current_prompt = Prompt(content=initial_prompt)
        current_result = await self._evaluate_prompt(current_prompt, validation_cases)

        best_prompt = current_prompt.copy()
        best_score = current_result.score
        history = [current_result]

        bootstrapped_examples: List[Dict[str, Any]] = []
        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            self._log(
                f"Iteration {iteration + 1}/{self.max_iterations} - "
                f"Best score: {best_score:.4f}, "
                f"Examples: {len(bootstrapped_examples)}"
            )

            # Generate more examples from bootstrap cases
            new_examples = await self._bootstrap_examples(
                current_prompt,
                bootstrap_cases,
                max_new=3
            )

            if new_examples:
                bootstrapped_examples.extend(new_examples)
                bootstrapped_examples = bootstrapped_examples[:self.max_bootstrapped_examples]

                self._log(f"Bootstrapped {len(new_examples)} new examples")

                # Build few-shot prompt with examples
                few_shot_prompt = self._build_few_shot_prompt(
                    initial_prompt,
                    bootstrapped_examples
                )

                # Evaluate
                result = await self._evaluate_prompt(few_shot_prompt, validation_cases)
                history.append(result)

                if result.score > best_score:
                    best_score = result.score
                    best_prompt = few_shot_prompt.copy()
                    current_prompt = few_shot_prompt
                    no_improvement_count = 0
                    self._log(f"New best score: {best_score:.4f}")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                self._log("No new examples bootstrapped")

            # Check convergence
            if no_improvement_count >= self.patience:
                self._log(f"Converged after {iteration + 1} iterations")
                break

        end_time = datetime.now()
        self._log(
            f"Optimization complete. Best score: {best_score:.4f}, "
            f"Total examples: {len(bootstrapped_examples)}"
        )

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            history=history,
            metadata={
                "iterations_run": iteration + 1,
                "total_evaluations": len(history),
                "bootstrapped_examples": bootstrapped_examples,
                "num_examples": len(bootstrapped_examples),
            },
            start_time=start_time,
            end_time=end_time,
        )

    async def _bootstrap_examples(
        self,
        prompt: Prompt,
        cases: List[Dict[str, Any]],
        max_new: int = 3
    ) -> List[Dict[str, Any]]:
        """Bootstrap examples from successful predictions."""
        new_examples = []

        # Shuffle cases to get variety
        import random
        sampled_cases = random.sample(cases, min(max_new * 2, len(cases)))

        for case in sampled_cases:
            if len(new_examples) >= max_new:
                break

            # Generate prediction
            rendered = prompt.render(**case) if prompt.variables else prompt.content

            # Add case inputs to prompt
            if "{" in rendered:
                # Template-based
                test_prompt = rendered
                for key, value in case.items():
                    test_prompt = test_prompt.replace(f"{{{key}}}", str(value))
            else:
                # Append inputs
                test_prompt = f"{rendered}\n\nInput: {case}"

            response = await self.provider.complete(test_prompt, temperature=0)
            prediction = response.content.strip()

            # Check if this is a successful prediction
            is_success = self._check_success(prediction, case)

            if is_success:
                # Create example
                example = {
                    **case,
                    "output": prediction,
                }
                new_examples.append(example)
                self._log(f"  Bootstrapped example from: {case}")

        return new_examples

    def _check_success(self, prediction: str, case: Dict[str, Any]) -> bool:
        """Check if a prediction is successful."""
        if "expected" in case or "expected_output" in case:
            expected = case.get("expected") or case.get("expected_output")
            # Simple check: expected in prediction
            return str(expected).lower() in prediction.lower()
        # If no expected output, assume success (or could use other heuristics)
        return len(prediction) > 0

    def _build_few_shot_prompt(
        self,
        template: str,
        examples: List[Dict[str, Any]]
    ) -> Prompt:
        """Build a few-shot prompt with examples."""
        if not examples:
            return Prompt(content=template)

        # Format examples
        examples_str = "\n\n".join([
            self._format_example(ex, i + 1)
            for i, ex in enumerate(examples)
        ])

        # Insert examples into template
        if "{examples}" in template:
            content = template.replace("{examples}", examples_str)
        else:
            # Prepend examples
            content = f"{examples_str}\n\n{template}"

        return Prompt(
            content=content,
            metadata={"examples": examples, "num_examples": len(examples)}
        )

    def _format_example(self, example: Dict[str, Any], index: int) -> str:
        """Format a single example."""
        parts = [f"Example {index}:"]

        # Separate output from inputs
        output = example.pop("output", None)
        expected = example.pop("expected", None) or example.pop("expected_output", None)

        # Format inputs
        for key, value in example.items():
            if key not in ["output", "expected", "expected_output"]:
                parts.append(f"{key}: {value}")

        # Add output
        if output:
            parts.append(f"Output: {output}")
        elif expected:
            parts.append(f"Output: {expected}")

        return "\n".join(parts)
