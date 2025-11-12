"""Meta-prompt optimizer (Arize-inspired).

This optimizer uses an LLM to reflect on prompts and generate improved versions
based on evaluation feedback.
"""

from typing import Any, Dict, List
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.providers.base import LLMProvider


class MetaPromptOptimizer(Optimizer):
    """Meta-prompt optimizer that uses LLM reflection to improve prompts.

    Inspired by Arize Phoenix's meta-prompting approach.
    """

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        max_iterations: int = 20,
        patience: int = 5,
        verbose: bool = True,
        meta_temperature: float = 0.8,
    ):
        """Initialize meta-prompt optimizer.

        Args:
            provider: LLM provider for both evaluation and meta-prompting
            evaluator: Evaluator for scoring prompts
            max_iterations: Maximum optimization iterations
            patience: Early stopping patience
            verbose: Whether to print progress
            meta_temperature: Temperature for meta-prompt generation
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.meta_temperature = meta_temperature

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize using meta-prompting.

        Args:
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and history
        """
        start_time = datetime.now()
        self._log("Starting meta-prompt optimization")

        current_prompt = Prompt(content=initial_prompt)
        current_result = await self._evaluate_prompt(current_prompt, test_cases)

        best_prompt = current_prompt.copy()
        best_score = current_result.score
        history = [current_result]

        # Collect feedback from test cases
        feedback_history: List[Dict[str, Any]] = []

        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            self._log(
                f"Iteration {iteration + 1}/{self.max_iterations} - "
                f"Current score: {current_result.score:.4f}"
            )

            # Collect detailed feedback by running test cases
            iteration_feedback = await self._collect_feedback(
                current_prompt, test_cases
            )
            feedback_history.append(iteration_feedback)

            # Generate improved prompt using meta-prompting
            new_prompt = await self._generate_improved_prompt(
                current_prompt,
                current_result,
                iteration_feedback,
                test_cases
            )

            # Evaluate new prompt
            new_result = await self._evaluate_prompt(new_prompt, test_cases)
            history.append(new_result)

            # Update best
            if new_result.score > best_score:
                best_score = new_result.score
                best_prompt = new_prompt.copy()
                no_improvement_count = 0
                self._log(f"New best score: {best_score:.4f}")
            else:
                no_improvement_count += 1

            # Update current if improved
            if new_result.score > current_result.score:
                current_prompt = new_prompt
                current_result = new_result
            else:
                self._log(f"No improvement, keeping current prompt")

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
                "iterations_run": iteration + 1,
                "total_evaluations": len(history),
                "feedback_collected": len(feedback_history),
            },
            start_time=start_time,
            end_time=end_time,
        )

    async def _collect_feedback(
        self,
        prompt: Prompt,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collect feedback by running test cases and analyzing results."""
        successes = []
        failures = []

        # Sample a subset of test cases for feedback
        sample_size = min(5, len(test_cases))
        import random
        sampled_cases = random.sample(test_cases, sample_size)

        for test_case in sampled_cases:
            # Render prompt with test case
            rendered = prompt.render(**test_case) if prompt.variables else prompt.content

            # Get LLM response
            response = await self.provider.complete(rendered, temperature=0)

            # Check if this is a success or failure
            is_success = self._check_success(response.content, test_case)

            if is_success:
                successes.append({
                    "input": test_case,
                    "output": response.content,
                })
            else:
                failures.append({
                    "input": test_case,
                    "output": response.content,
                    "expected": test_case.get("expected", "N/A"),
                })

        return {
            "successes": successes,
            "failures": failures,
            "success_rate": len(successes) / len(sampled_cases) if sampled_cases else 0,
        }

    def _check_success(self, output: str, test_case: Dict[str, Any]) -> bool:
        """Check if an output is successful."""
        if "expected" in test_case or "expected_output" in test_case:
            expected = test_case.get("expected") or test_case.get("expected_output")
            return str(expected).lower() in output.lower()
        return True  # Assume success if no expected output

    async def _generate_improved_prompt(
        self,
        current_prompt: Prompt,
        current_result: EvaluationResult,
        feedback: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Prompt:
        """Generate an improved prompt using meta-prompting."""
        meta_prompt = f"""You are a prompt optimization expert. Your task is to improve the following prompt based on evaluation feedback.

CURRENT PROMPT:
{current_prompt.content}

CURRENT PERFORMANCE:
- Score: {current_result.score:.4f}
- Metrics: {current_result.metrics}

FEEDBACK FROM TEST CASES:
- Success Rate: {feedback['success_rate']:.1%}
- Number of Successes: {len(feedback['successes'])}
- Number of Failures: {len(feedback['failures'])}

SUCCESSFUL CASES:
{self._format_cases(feedback['successes'][:2])}

FAILED CASES:
{self._format_cases(feedback['failures'][:3])}

ANALYSIS:
Based on the feedback above, identify:
1. What is the prompt doing well?
2. What are the main failure patterns?
3. What specific changes would address these failures?

IMPROVED PROMPT:
Generate an improved version of the prompt that:
- Maintains what is working well
- Addresses the identified failure patterns
- Is clear, specific, and effective
- Follows best practices for prompt engineering

Return ONLY the improved prompt text, without any explanation or meta-commentary:"""

        response = await self.provider.complete(
            meta_prompt,
            temperature=self.meta_temperature,
            max_tokens=2000
        )

        improved_content = response.content.strip()

        # Clean up any markdown code blocks or explanations
        if "```" in improved_content:
            # Extract content from code blocks
            parts = improved_content.split("```")
            for part in parts:
                if part.strip() and not part.strip().startswith(("python", "text", "markdown")):
                    improved_content = part.strip()
                    break

        return Prompt(content=improved_content)

    def _format_cases(self, cases: List[Dict[str, Any]]) -> str:
        """Format cases for display in meta-prompt."""
        if not cases:
            return "None"

        formatted = []
        for i, case in enumerate(cases, 1):
            formatted.append(f"\nCase {i}:")
            formatted.append(f"  Input: {case.get('input', {})}")
            formatted.append(f"  Output: {case.get('output', 'N/A')[:200]}")
            if 'expected' in case:
                formatted.append(f"  Expected: {case['expected']}")

        return "\n".join(formatted)
