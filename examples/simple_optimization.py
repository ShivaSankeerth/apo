"""Simple example of using APO for prompt optimization."""

import asyncio
from apo import GeneticOptimizer, AnthropicProvider
from apo.core import Prompt, Evaluator, EvaluationResult


class SimpleSentimentEvaluator(Evaluator):
    """Simple evaluator for sentiment classification."""

    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    async def evaluate(self, prompt, test_cases):
        """Evaluate prompt on sentiment classification task."""
        correct = 0
        total = len(test_cases)

        for test_case in test_cases:
            # Render prompt with test input
            full_prompt = f"{prompt.content}\n\nText: {test_case['text']}\nSentiment:"

            # Get LLM response
            response = await self.provider.complete(full_prompt, temperature=0)
            prediction = response.content.strip().lower()

            # Check if correct
            expected = test_case['sentiment'].lower()
            if expected in prediction or prediction in expected:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": accuracy},
            metadata={"correct": correct, "total": total}
        )


async def main():
    # Set up provider (replace with your API key)
    provider = AnthropicProvider(
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"
    )

    # Define test cases
    test_cases = [
        {"text": "I love this product! It's amazing!", "sentiment": "positive"},
        {"text": "This is terrible. Worst purchase ever.", "sentiment": "negative"},
        {"text": "It's okay, nothing special.", "sentiment": "neutral"},
        {"text": "Absolutely fantastic experience!", "sentiment": "positive"},
        {"text": "Very disappointed with the quality.", "sentiment": "negative"},
    ]

    # Create evaluator
    evaluator = SimpleSentimentEvaluator(provider)

    # Set up optimizer
    optimizer = GeneticOptimizer(
        provider=provider,
        evaluator=evaluator,
        population_size=5,
        generations=3,
        verbose=True
    )

    # Initial prompt
    initial_prompt = "Classify the sentiment of the following text as positive, negative, or neutral."

    # Run optimization
    print("Starting optimization...")
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        test_cases=test_cases
    )

    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"Total Evaluations: {result.total_evaluations}")
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"\nBest Prompt:\n{result.best_prompt.content}")

    # Show top 3 prompts
    print("\n" + "="*50)
    print("TOP 3 PROMPTS")
    print("="*50)
    for i, eval_result in enumerate(result.get_best_n(3), 1):
        print(f"\n{i}. Score: {eval_result.score:.4f}")
        print(f"   Prompt: {eval_result.prompt.content[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())
