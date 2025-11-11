"""Genetic algorithm for prompt optimization."""

import random
from typing import Any, Dict, List
from datetime import datetime
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.prompt import Prompt
from apo.core.evaluator import Evaluator
from apo.providers.base import LLMProvider


class GeneticOptimizer(Optimizer):
    """Genetic algorithm-based prompt optimizer."""

    def __init__(
        self,
        provider: LLMProvider,
        evaluator: Evaluator,
        population_size: int = 10,
        generations: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_size: int = 2,
        max_iterations: int = 100,
        patience: int = 10,
        verbose: bool = True,
    ):
        """Initialize genetic optimizer.

        Args:
            provider: LLM provider for generating variations
            evaluator: Evaluator for scoring prompts
            population_size: Number of prompts in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top prompts to preserve
            max_iterations: Maximum iterations
            patience: Iterations without improvement before stopping
            verbose: Whether to print progress
        """
        super().__init__(provider, evaluator, max_iterations, patience, verbose)
        self.population_size = population_size
        self.generations = min(generations, max_iterations)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

    async def optimize(
        self,
        initial_prompt: str,
        test_cases: List[Dict[str, Any]],
        **kwargs: Any
    ) -> OptimizationResult:
        """Optimize using genetic algorithm.

        Args:
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            **kwargs: Additional parameters

        Returns:
            OptimizationResult with best prompt and history
        """
        start_time = datetime.now()
        self._log(f"Starting genetic optimization with population={self.population_size}, generations={self.generations}")

        # Initialize population
        population = await self._initialize_population(initial_prompt)
        history = []
        best_prompt = None
        best_score = float('-inf')
        no_improvement_count = 0

        for generation in range(self.generations):
            self._log(f"Generation {generation + 1}/{self.generations}")

            # Evaluate population
            evaluated = []
            for prompt in population:
                result = await self._evaluate_prompt(prompt, test_cases)
                evaluated.append(result)
                history.append(result)

                if result.score > best_score:
                    best_score = result.score
                    best_prompt = prompt.copy()
                    no_improvement_count = 0
                    self._log(f"New best score: {best_score:.4f}")
                else:
                    no_improvement_count += 1

            # Check for convergence
            if no_improvement_count >= self.patience:
                self._log(f"Converged after {generation + 1} generations")
                break

            # Selection and evolution
            population = await self._evolve_population(evaluated)

        end_time = datetime.now()

        if best_prompt is None:
            best_prompt = Prompt(content=initial_prompt)

        self._log(f"Optimization complete. Best score: {best_score:.4f}")

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            history=history,
            metadata={
                "population_size": self.population_size,
                "generations_run": generation + 1,
                "total_evaluations": len(history),
            },
            start_time=start_time,
            end_time=end_time,
        )

    async def _initialize_population(self, initial_prompt: str) -> List[Prompt]:
        """Create initial population with variations of the prompt."""
        self._log("Initializing population...")

        variations = await self.provider.generate_prompt_variations(
            initial_prompt,
            num_variations=self.population_size - 1
        )

        population = [Prompt(content=initial_prompt)]
        for variation in variations:
            population.append(Prompt(content=variation))

        return population

    async def _evolve_population(self, evaluated: List) -> List[Prompt]:
        """Evolve the population through selection, crossover, and mutation."""
        # Sort by score
        evaluated.sort(key=lambda x: x.score, reverse=True)

        # Elitism: keep top performers
        new_population = [result.prompt.copy() for result in evaluated[:self.elite_size]]

        # Generate rest of population
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(evaluated)
            parent2 = self._tournament_selection(evaluated)

            # Crossover
            if random.random() < self.crossover_rate:
                child = await self._crossover(parent1, parent2)
            else:
                child = parent1.copy()

            # Mutation
            if random.random() < self.mutation_rate:
                child = await self._mutate(child)

            new_population.append(child)

        return new_population

    def _tournament_selection(self, evaluated: List, tournament_size: int = 3) -> Prompt:
        """Select a prompt using tournament selection."""
        tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
        winner = max(tournament, key=lambda x: x.score)
        return winner.prompt

    async def _crossover(self, parent1: Prompt, parent2: Prompt) -> Prompt:
        """Combine two prompts through crossover."""
        crossover_prompt = f"""Combine these two prompts into a single prompt that takes the best elements from both:

Prompt 1: {parent1.content}

Prompt 2: {parent2.content}

Create a new prompt that combines their strengths. Return only the new prompt without explanation:"""

        response = await self.provider.complete(crossover_prompt, temperature=0.7)
        return Prompt(content=response.content.strip())

    async def _mutate(self, prompt: Prompt) -> Prompt:
        """Mutate a prompt by making small variations."""
        mutation_prompt = f"""Make a small variation to this prompt. Keep the core meaning but change the phrasing, add clarity, or adjust the structure slightly:

{prompt.content}

Return only the modified prompt without explanation:"""

        response = await self.provider.complete(mutation_prompt, temperature=0.8)
        return Prompt(content=response.content.strip())
