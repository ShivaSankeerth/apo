"""Example using BootstrapOptimizer (DSPy-inspired)."""

import asyncio
from apo import BootstrapOptimizer, AnthropicProvider
from apo.core import Evaluator, EvaluationResult


class SimpleClassificationEvaluator(Evaluator):
    """Simple evaluator for classification."""

    async def evaluate(self, prompt, test_cases):
        """Evaluate classification accuracy."""
        # For this example, we'll use a simple heuristic
        # In practice, you'd call the LLM with the prompt
        score = 0.7  # Placeholder
        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": score}
        )


async def main():
    # Set up provider
    provider = AnthropicProvider(
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"
    )

    # Test cases with inputs and expected outputs
    test_cases = [
        {
            "text": "This product is amazing! Best purchase ever!",
            "expected": "positive"
        },
        {
            "text": "Terrible quality. Very disappointed.",
            "expected": "negative"
        },
        {
            "text": "It's okay. Nothing special.",
            "expected": "neutral"
        },
        {
            "text": "Absolutely love it! Highly recommend!",
            "expected": "positive"
        },
        {
            "text": "Waste of money. Don't buy.",
            "expected": "negative"
        },
        {
            "text": "Average product. Works as expected.",
            "expected": "neutral"
        },
    ]

    # Create evaluator
    evaluator = SimpleClassificationEvaluator()

    # Set up bootstrap optimizer
    optimizer = BootstrapOptimizer(
        provider=provider,
        evaluator=evaluator,
        max_bootstrapped_examples=8,
        max_iterations=5,
        patience=2,
        verbose=True
    )

    # Initial prompt template (with {examples} placeholder)
    initial_prompt = """Classify the sentiment of text as positive, negative, or neutral.

{examples}

Now classify this text:
Text: {text}
Sentiment:"""

    # Run optimization
    print("Starting bootstrap optimization...")
    print("The optimizer will automatically generate few-shot examples from successful predictions.\n")

    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        test_cases=test_cases
    )

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nBest Score: {result.best_score:.4f}")
    print(f"Bootstrapped Examples: {result.metadata.get('num_examples', 0)}")
    print(f"Total Evaluations: {result.total_evaluations}")

    print(f"\nFinal Few-Shot Prompt:\n{result.best_prompt.content[:500]}...")

    # Show bootstrapped examples
    if "bootstrapped_examples" in result.metadata:
        print("\n" + "="*60)
        print("BOOTSTRAPPED EXAMPLES")
        print("="*60)
        for i, example in enumerate(result.metadata["bootstrapped_examples"][:3], 1):
            print(f"\nExample {i}:")
            print(f"  Text: {example.get('text', 'N/A')}")
            print(f"  Output: {example.get('output', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
