"""Generate demo benchmark results for visualization testing.

This creates mock results without needing to run actual API calls.
"""

import json
import random
from datetime import datetime
from benchmarks.runner import BenchmarkResult
from benchmarks.visualize import generate_all_visualizations


def generate_mock_results():
    """Generate realistic mock benchmark results."""
    optimizers = [
        "Random Search",
        "Hill Climbing",
        "Simulated Annealing",
        "Genetic Algorithm",
        "Meta-Prompt",
        "Reflection",
        "Reflection+Pareto",
        "Bootstrap"
    ]

    tasks = [
        "Sentiment Classification",
        "Question Answering",
        "Category Classification",
        "Instruction Following"
    ]

    # Base performance characteristics for each optimizer
    optimizer_traits = {
        "Random Search": {"improvement": 0.10, "speed": 1.0, "variance": 0.05},
        "Hill Climbing": {"improvement": 0.15, "speed": 0.8, "variance": 0.06},
        "Simulated Annealing": {"improvement": 0.18, "speed": 0.7, "variance": 0.07},
        "Genetic Algorithm": {"improvement": 0.20, "speed": 0.6, "variance": 0.08},
        "Meta-Prompt": {"improvement": 0.28, "speed": 1.2, "variance": 0.04},
        "Reflection": {"improvement": 0.25, "speed": 0.5, "variance": 0.05},
        "Reflection+Pareto": {"improvement": 0.30, "speed": 0.4, "variance": 0.06},
        "Bootstrap": {"improvement": 0.22, "speed": 0.9, "variance": 0.07},
    }

    results = []

    for task in tasks:
        # Base initial score varies by task
        base_initial = random.uniform(0.5, 0.7)

        for optimizer in optimizers:
            traits = optimizer_traits[optimizer]

            # Add task-specific and random variations
            initial_score = base_initial + random.gauss(0, 0.05)
            initial_score = max(0.3, min(0.8, initial_score))

            # Calculate improvement with variance
            improvement = traits["improvement"] + random.gauss(0, traits["variance"])
            improvement = max(0.0, min(0.35, improvement))

            final_score = min(1.0, initial_score + improvement)

            # Duration based on speed trait
            base_duration = 20.0
            duration = base_duration / traits["speed"] * random.uniform(0.8, 1.2)

            # Evaluations correlate with duration
            evaluations = int(duration * random.uniform(2, 5))

            # Iterations
            if "Random" in optimizer:
                iterations = evaluations
            else:
                iterations = int(evaluations / random.uniform(3, 8))

            # Generate score history (convergence curve)
            num_points = iterations
            score_history = []
            current_score = initial_score

            for i in range(num_points):
                progress = i / num_points
                # Logarithmic improvement curve
                target_improvement = improvement * (1 - (1 - progress) ** 2)
                current_score = initial_score + target_improvement + random.gauss(0, 0.02)
                current_score = min(final_score, max(initial_score, current_score))
                score_history.append(current_score)

            # Find convergence point
            convergence_iteration = None
            for i, score in enumerate(score_history):
                if score >= final_score * 0.99:
                    convergence_iteration = i
                    break

            result = BenchmarkResult(
                optimizer_name=optimizer,
                task_name=task,
                initial_score=initial_score,
                final_score=final_score,
                improvement=improvement,
                duration_seconds=duration,
                total_evaluations=evaluations,
                iterations=iterations,
                convergence_iteration=convergence_iteration,
                score_history=score_history,
                best_prompt=f"Optimized prompt for {task} by {optimizer}",
                metadata={}
            )

            results.append(result)

    return results


if __name__ == "__main__":
    print("Generating mock benchmark results...")
    results = generate_mock_results()

    print(f"Generated {len(results)} results")
    print("\nGenerating visualizations...")

    # Save results
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
            for r in results
        ]
    }

    with open("benchmark_results_demo/results.json", "w") as f:
        json.dump(data, f, indent=2)

    # Generate all visualizations
    generate_all_visualizations(results, "benchmark_results_demo")

    print("\n✅ Demo results generated!")
    print("📊 View visualizations in: benchmark_results_demo/")
    print("\nKey files:")
    print("  - performance_comparison.html")
    print("  - efficiency_analysis.html")
    print("  - winners.html")
    print("  - convergence_*.html")
