"""Reflection-based optimizer with Pareto frontier (GEPA-inspired).

This optimizer uses LLM reflection to analyze what works and what doesn't,
then evolves prompts accordingly. It maintains a Pareto frontier of
complementary strategies.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.core.pareto import ParetoFrontier
from apo.providers.base import LLMProvider


class ReflectionOptimizer(Optimizer):
    """Reflection-based optimizer inspired by DSPy GEPA.

    Uses LLM reflection to understand what works and evolves prompts
    using a tree-based search with Pareto frontier tracking.
    """

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        max_iterations: int = 30,
        patience: int = 10,
        use_pareto_frontier: bool = True,
        population_size: int = 5,
        verbose: bool = True,
        reflection_temperature: float = 0.7,
    ):
        """Initialize reflection optimizer.

        Args:
            provider: LLM provider for reflection and generation
            evaluator: Evaluator for scoring prompts
            max_iterations: Maximum optimization iterations
            patience: Early stopping patience
            use_pareto_frontier: Whether to use Pareto frontier tracking
            population_size: Size of candidate pool
            verbose: Whether to print progress
            reflection_temperature: Temperature for reflection generation
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.use_pareto_frontier = use_pareto_frontier
        self.population_size = population_size
        self.reflection_temperature = reflection_temperature
        self.pareto_frontier: Optional[ParetoFrontier] = None

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize using reflection and Pareto frontier.

        Args:
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and history
        """
        start_time = datetime.now()
        mode = "Pareto frontier" if self.use_pareto_frontier else "single best"
        self._log(f"Starting reflection-based optimization with {mode}")

        if self.use_pareto_frontier:
            self.pareto_frontier = ParetoFrontier()

        # Evaluate initial prompt
        initial = Prompt(content=initial_prompt)
        initial_result = await self._evaluate_prompt(initial, test_cases)

        best_prompt = initial.copy()
        best_score = initial_result.score
        history = [initial_result]

        if self.use_pareto_frontier:
            self.pareto_frontier.add_candidate(initial_result)

        # Track evolution tree
        evolution_tree: List[Dict[str, Any]] = []

        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            self._log(
                f"Iteration {iteration + 1}/{self.max_iterations} - "
                f"Best score: {best_score:.4f}"
            )

            if self.use_pareto_frontier:
                self._log(f"Pareto frontier size: {self.pareto_frontier.size()}")

            # Select parent prompt
            if self.use_pareto_frontier and self.pareto_frontier.size() > 0:
                parent_result = self.pareto_frontier.get_random_candidate()
                parent_prompt = parent_result.prompt
            else:
                parent_prompt = best_prompt

            # Generate reflection on current prompt
            reflection = await self._generate_reflection(
                parent_prompt,
                test_cases,
                history
            )

            # Generate evolved prompts based on reflection
            evolved_prompts = await self._evolve_from_reflection(
                parent_prompt,
                reflection,
                num_variations=self.population_size
            )

            # Evaluate evolved prompts
            improved = False
            for evolved in evolved_prompts:
                result = await self._evaluate_prompt(evolved, test_cases)
                history.append(result)

                # Track in evolution tree
                evolution_tree.append({
                    "iteration": iteration,
                    "parent": parent_prompt.content[:50],
                    "child": evolved.content[:50],
                    "reflection": reflection[:100],
                    "score": result.score,
                })

                # Update Pareto frontier
                if self.use_pareto_frontier:
                    self.pareto_frontier.add_candidate(result)

                # Update best
                if result.score > best_score:
                    best_score = result.score
                    best_prompt = evolved.copy()
                    improved = True
                    no_improvement_count = 0
                    self._log(f"New best score: {best_score:.4f}")

            if not improved:
                no_improvement_count += 1

            # Check convergence
            if no_improvement_count >= self.patience:
                self._log(f"No improvement for {self.patience} iterations. Stopping.")
                break

        end_time = datetime.now()
        self._log(f"Optimization complete. Best score: {best_score:.4f}")

        metadata = {
            "iterations_run": iteration + 1,
            "total_evaluations": len(history),
            "evolution_tree_size": len(evolution_tree),
        }

        if self.use_pareto_frontier and self.pareto_frontier:
            metadata["pareto_frontier"] = self.pareto_frontier.summary()

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            history=history,
            metadata=metadata,
            start_time=start_time,
            end_time=end_time,
        )

    async def _generate_reflection(
        self,
        prompt: Prompt,
        test_cases: List[Dict[str, Any]],
        history: List[EvaluationResult]
    ) -> str:
        """Generate reflection on what works and what doesn't."""
        # Sample test cases
        import random
        sample_size = min(3, len(test_cases))
        sampled = random.sample(test_cases, sample_size)

        # Collect some example outputs
        examples = []
        for test_case in sampled:
            rendered = prompt.render(**test_case) if prompt.variables else prompt.content
            response = await self.provider.complete(rendered, temperature=0)
            examples.append({
                "input": test_case,
                "output": response.content[:200],
            })

        # Get recent performance trends
        recent_scores = [r.score for r in history[-5:]] if len(history) >= 5 else [r.score for r in history]
        trend = "improving" if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else "stagnant"

        reflection_prompt = f"""You are a prompt engineering expert analyzing a prompt's performance.

CURRENT PROMPT:
{prompt.content}

RECENT PERFORMANCE TREND: {trend}
RECENT SCORES: {[f"{s:.2f}" for s in recent_scores]}

EXAMPLE OUTPUTS:
{self._format_examples(examples)}

REFLECTION TASK:
Analyze this prompt and provide a structured reflection:

1. WHAT'S WORKING WELL:
   - What aspects of the prompt are effective?
   - What patterns lead to good outputs?

2. WHAT'S NOT WORKING:
   - What are the main weaknesses?
   - What failure patterns do you observe?
   - What edge cases is it missing?

3. IMPROVEMENT DIRECTIONS:
   - What specific changes would address the weaknesses?
   - What additional context or constraints would help?
   - How can we make it more robust?

Provide your reflection in a clear, actionable format:"""

        response = await self.provider.complete(
            reflection_prompt,
            temperature=self.reflection_temperature,
            max_tokens=1500
        )

        return response.content

    async def _evolve_from_reflection(
        self,
        prompt: Prompt,
        reflection: str,
        num_variations: int = 3
    ) -> List[Prompt]:
        """Generate evolved prompts based on reflection."""
        evolution_prompt = f"""Based on the following reflection, generate {num_variations} improved versions of the prompt.

ORIGINAL PROMPT:
{prompt.content}

REFLECTION:
{reflection}

TASK:
Generate {num_variations} different improved versions that:
1. Address the identified weaknesses
2. Preserve what works well
3. Explore different improvement directions
4. Are concrete and ready to use

Format your response as:

VERSION 1:
[improved prompt 1]

VERSION 2:
[improved prompt 2]

VERSION 3:
[improved prompt 3]

Generate the improved prompts:"""

        response = await self.provider.complete(
            evolution_prompt,
            temperature=0.8,
            max_tokens=2000
        )

        # Parse evolved prompts
        evolved = []
        content = response.content

        # Split by VERSION markers
        import re
        versions = re.split(r'VERSION \d+:', content)

        for version_text in versions[1:]:  # Skip first empty split
            cleaned = version_text.strip()
            if cleaned:
                evolved.append(Prompt(content=cleaned))

        # If parsing failed, generate variations one by one
        if len(evolved) < num_variations:
            for i in range(num_variations - len(evolved)):
                variation = await self.provider.generate_prompt_variations(
                    prompt.content,
                    num_variations=1,
                    instructions=f"Improve based on: {reflection[:200]}"
                )
                if variation:
                    evolved.append(Prompt(content=variation[0]))

        return evolved[:num_variations]

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples for display."""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"\nExample {i}:")
            formatted.append(f"  Input: {ex['input']}")
            formatted.append(f"  Output: {ex['output']}")
        return "\n".join(formatted)

    def get_pareto_frontier(self) -> Optional[ParetoFrontier]:
        """Get the Pareto frontier if it was used."""
        return self.pareto_frontier
