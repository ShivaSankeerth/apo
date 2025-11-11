"""Example of multi-objective optimization."""

import asyncio
from apo import HillClimbingOptimizer, OpenAIProvider
from apo.core import Prompt, Evaluator, EvaluationResult
from apo.core.evaluator import MultiObjectiveEvaluator


class AccuracyEvaluator(Evaluator):
    """Evaluates accuracy."""

    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    async def evaluate(self, prompt, test_cases):
        correct = 0
        for test_case in test_cases:
            full_prompt = prompt.render(input=test_case['input'])
            response = await self.provider.complete(full_prompt, temperature=0)

            if test_case['expected'] in response.content:
                correct += 1

        accuracy = correct / len(test_cases) if test_cases else 0
        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": accuracy}
        )


class EfficiencyEvaluator(Evaluator):
    """Evaluates prompt efficiency (shorter is better)."""

    async def evaluate(self, prompt, test_cases):
        # Score based on prompt length (normalize to 0-1)
        max_length = 500
        length = len(prompt.content)
        efficiency = max(0, 1 - (length / max_length))

        return EvaluationResult(
            prompt=prompt,
            metrics={"efficiency": efficiency},
            metadata={"length": length}
        )


async def main():
    # Set up provider
    provider = OpenAIProvider(
        api_key="your-api-key-here",
        model="gpt-4-turbo-preview"
    )

    # Create multi-objective evaluator
    evaluator = MultiObjectiveEvaluator(
        evaluators=[
            AccuracyEvaluator(provider),
            EfficiencyEvaluator(provider)
        ],
        metric_weights={
            "accuracy": 0.7,  # 70% weight on accuracy
            "efficiency": 0.3  # 30% weight on efficiency
        }
    )

    # Set up optimizer
    optimizer = HillClimbingOptimizer(
        provider=provider,
        evaluator=evaluator,
        num_neighbors=3,
        max_iterations=10,
        verbose=True
    )

    # Test cases
    test_cases = [
        {"input": "2 + 2", "expected": "4"},
        {"input": "10 - 3", "expected": "7"},
        {"input": "5 * 6", "expected": "30"},
    ]

    # Initial prompt
    initial_prompt = "Solve the following math problem: {input}"

    # Run optimization
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        test_cases=test_cases
    )

    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"Best Prompt: {result.best_prompt.content}")


if __name__ == "__main__":
    asyncio.run(main())
