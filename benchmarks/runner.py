"""Benchmark runner for comparing optimizers."""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from apo import (
    GeneticOptimizer,
    HillClimbingOptimizer,
    SimulatedAnnealingOptimizer,
    RandomSearchOptimizer,
    MetaPromptOptimizer,
    ReflectionOptimizer,
    FewShotOptimizer,
    BootstrapOptimizer,
    AnthropicProvider,
    OpenAIProvider,
)
from apo.core import Evaluator, EvaluationResult, Prompt
from benchmarks.tasks import BenchmarkTask


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    optimizer_name: str
    task_name: str
    initial_score: float
    final_score: float
    improvement: float
    duration_seconds: float
    total_evaluations: int
    iterations: int
    convergence_iteration: Optional[int]
    score_history: List[float] = field(default_factory=list)
    best_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskEvaluator(Evaluator):
    """Evaluator for benchmark tasks."""

    def __init__(self, provider, task: BenchmarkTask):
        super().__init__()
        self.provider = provider
        self.task = task

    async def evaluate(self, prompt: Prompt, test_cases: List[Dict[str, Any]]) -> EvaluationResult:
        """Evaluate prompt on task."""
        scores = []

        for test_case in test_cases:
            # Render prompt
            rendered = prompt.content
            for key, value in test_case.items():
                if key != "expected":
                    rendered = rendered.replace(f"{{{key}}}", str(value))

            # Get LLM response
            try:
                response = await self.provider.complete(rendered, temperature=0, max_tokens=100)
                output = response.content.strip()
            except Exception as e:
                print(f"Error getting response: {e}")
                output = ""

            # Score against expected
            expected = test_case.get("expected", "")
            score = self.task.evaluate_output(output, expected)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": avg_score},
            metadata={"scores": scores}
        )


class BenchmarkRunner:
    """Run benchmarks across multiple optimizers and tasks."""

    def __init__(self, provider, verbose: bool = True):
        self.provider = provider
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[Benchmark] {message}")

    async def run_optimizer(
        self,
        optimizer_class,
        optimizer_name: str,
        task: BenchmarkTask,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Run a single optimizer on a task."""
        self.log(f"Running {optimizer_name} on {task.name}...")

        # Create evaluator
        evaluator = TaskEvaluator(self.provider, task)

        # Create optimizer
        kwargs = optimizer_kwargs or {}
        kwargs.update({
            "provider": self.provider,
            "evaluator": evaluator,
            "verbose": False
        })

        optimizer = optimizer_class(**kwargs)

        # Measure initial performance
        initial_prompt = Prompt(content=task.initial_prompt)
        initial_result = await evaluator.evaluate(initial_prompt, task.test_cases)
        initial_score = initial_result.score

        # Run optimization
        start_time = time.time()
        try:
            opt_result = await optimizer.optimize(
                initial_prompt=task.initial_prompt,
                test_cases=task.test_cases
            )
            duration = time.time() - start_time

            # Extract metrics
            final_score = opt_result.best_score
            improvement = final_score - initial_score

            # Get score history
            score_history = [r.score for r in opt_result.history]

            # Find convergence point (first time we hit best score)
            convergence_iteration = None
            for i, score in enumerate(score_history):
                if score >= final_score * 0.99:  # Within 1% of best
                    convergence_iteration = i
                    break

            result = BenchmarkResult(
                optimizer_name=optimizer_name,
                task_name=task.name,
                initial_score=initial_score,
                final_score=final_score,
                improvement=improvement,
                duration_seconds=duration,
                total_evaluations=opt_result.total_evaluations,
                iterations=opt_result.metadata.get("iterations_run", len(score_history)),
                convergence_iteration=convergence_iteration,
                score_history=score_history,
                best_prompt=opt_result.best_prompt.content,
                metadata=opt_result.metadata
            )

            self.log(
                f"  ✓ {optimizer_name}: {initial_score:.2%} → {final_score:.2%} "
                f"(+{improvement:.2%}) in {duration:.1f}s"
            )

        except Exception as e:
            self.log(f"  ✗ {optimizer_name} failed: {str(e)}")
            result = BenchmarkResult(
                optimizer_name=optimizer_name,
                task_name=task.name,
                initial_score=initial_score,
                final_score=initial_score,
                improvement=0.0,
                duration_seconds=time.time() - start_time,
                total_evaluations=0,
                iterations=0,
                convergence_iteration=None,
                metadata={"error": str(e)}
            )

        self.results.append(result)
        return result

    async def run_all_optimizers(
        self,
        task: BenchmarkTask,
        timeout_per_optimizer: int = 300
    ) -> List[BenchmarkResult]:
        """Run all optimizers on a task."""
        self.log(f"\n{'='*60}")
        self.log(f"Benchmarking: {task.name}")
        self.log(f"{'='*60}")

        optimizers = [
            # Classic algorithms
            (RandomSearchOptimizer, "Random Search", {"num_samples": 20}),
            (HillClimbingOptimizer, "Hill Climbing", {"num_neighbors": 3, "max_iterations": 15}),
            (SimulatedAnnealingOptimizer, "Simulated Annealing", {"max_iterations": 20}),
            (GeneticOptimizer, "Genetic Algorithm", {"population_size": 8, "generations": 10}),
            # Advanced algorithms
            (MetaPromptOptimizer, "Meta-Prompt", {"max_iterations": 10, "patience": 3}),
            (ReflectionOptimizer, "Reflection", {"max_iterations": 10, "patience": 3, "use_pareto_frontier": False}),
            (ReflectionOptimizer, "Reflection+Pareto", {"max_iterations": 10, "patience": 3, "use_pareto_frontier": True}),
            (BootstrapOptimizer, "Bootstrap", {"max_iterations": 5, "patience": 2}),
        ]

        task_results = []
        for optimizer_class, name, kwargs in optimizers:
            result = await self.run_optimizer(optimizer_class, name, task, kwargs)
            task_results.append(result)

            # Small delay between optimizers
            await asyncio.sleep(1)

        return task_results

    async def run_benchmark_suite(
        self,
        tasks: List[BenchmarkTask]
    ) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        self.log(f"\n{'='*60}")
        self.log(f"APO Optimizer Benchmark Suite")
        self.log(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Tasks: {len(tasks)}")
        self.log(f"{'='*60}\n")

        all_results = []
        for task in tasks:
            task_results = await self.run_all_optimizers(task)
            all_results.extend(task_results)

        self.log(f"\n{'='*60}")
        self.log(f"Benchmark Complete!")
        self.log(f"Total runs: {len(all_results)}")
        self.log(f"{'='*60}\n")

        return all_results

    def save_results(self, filepath: str) -> None:
        """Save results to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "results": [
                {
                    "optimizer_name": r.optimizer_name,
                    "task_name": r.task_name,
                    "initial_score": r.initial_score,
                    "final_score": r.final_score,
                    "improvement": r.improvement,
                    "duration_seconds": r.duration_seconds,
                    "total_evaluations": r.total_evaluations,
                    "iterations": r.iterations,
                    "convergence_iteration": r.convergence_iteration,
                    "score_history": r.score_history,
                    "best_prompt": r.best_prompt,
                    "metadata": r.metadata,
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.log(f"Results saved to: {filepath}")
