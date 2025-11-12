"""Example using ReflectionOptimizer with Pareto frontier (GEPA-inspired)."""

import asyncio
from apo import ReflectionOptimizer, AnthropicProvider
from apo.core import Evaluator, EvaluationResult


class MultiCriteriaEvaluator(Evaluator):
    """Evaluator with multiple criteria (accuracy and conciseness)."""

    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    async def evaluate(self, prompt, test_cases):
        """Evaluate on multiple criteria."""
        correct = 0
        total_length = 0
        total = len(test_cases)

        for test_case in test_cases:
            full_prompt = f"{prompt.content}\n\nInput: {test_case['input']}"
            response = await self.provider.complete(full_prompt, temperature=0)
            prediction = response.content.strip()

            # Check accuracy
            if test_case['expected'].lower() in prediction.lower():
                correct += 1

            # Track length
            total_length += len(prediction)

        accuracy = correct / total if total > 0 else 0
        avg_length = total_length / total if total > 0 else 0

        # Conciseness score (prefer shorter responses)
        conciseness = max(0, 1 - (avg_length / 500))

        return EvaluationResult(
            prompt=prompt,
            metrics={
                "accuracy": accuracy,
                "conciseness": conciseness
            },
            metadata={"correct": correct, "avg_length": avg_length}
        )


async def main():
    # Set up provider
    provider = AnthropicProvider(
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"
    )

    # Test cases
    test_cases = [
        {"input": "Classify: I love this!", "expected": "positive"},
        {"input": "Classify: This is terrible.", "expected": "negative"},
        {"input": "Classify: It's okay.", "expected": "neutral"},
        {"input": "Classify: Amazing product!", "expected": "positive"},
        {"input": "Classify: Not good.", "expected": "negative"},
    ]

    # Create evaluator
    evaluator = MultiCriteriaEvaluator(provider)

    # Set up reflection optimizer with Pareto frontier
    optimizer = ReflectionOptimizer(
        provider=provider,
        evaluator=evaluator,
        use_pareto_frontier=True,
        population_size=3,
        max_iterations=15,
        patience=5,
        verbose=True
    )

    # Initial prompt
    initial_prompt = "Classify the sentiment as positive, negative, or neutral."

    # Run optimization
    print("Starting reflection-based optimization with Pareto frontier...")
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        test_cases=test_cases
    )

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"Total Evaluations: {result.total_evaluations}")
    print(f"\nBest Prompt:\n{result.best_prompt.content}")

    # Show Pareto frontier
    if optimizer.pareto_frontier:
        print("\n" + "="*60)
        print("PARETO FRONTIER")
        print("="*60)
        print(f"Frontier Size: {optimizer.pareto_frontier.size()}")
        print(f"\nSummary: {optimizer.pareto_frontier.summary()}")

        print("\nAll Pareto-optimal prompts:")
        for i, candidate in enumerate(optimizer.pareto_frontier.get_all_candidates(), 1):
            print(f"\n{i}. Metrics: {candidate.metrics}")
            print(f"   Prompt: {candidate.prompt.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
