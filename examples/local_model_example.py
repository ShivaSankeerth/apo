"""Example using Ollama for local prompt optimization.

This example shows how to optimize prompts using local models
without any API costs!

Prerequisites:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull llama2
3. pip install ollama
"""

import asyncio
from apo import HillClimbingOptimizer
from apo.providers import OllamaProvider
from apo.core import Evaluator, EvaluationResult, Prompt


class SimpleTaskEvaluator(Evaluator):
    """Simple evaluator for demonstration."""

    async def evaluate(self, prompt, test_cases):
        # For this example, we'll use a simple heuristic
        # In practice, you could use another local model to judge
        correct = 0

        for test_case in test_cases:
            # This would normally call the LLM, but for demo
            # we'll simulate it
            score = 0.7  # Placeholder
            correct += score

        avg_score = correct / len(test_cases) if test_cases else 0

        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": avg_score}
        )


async def main():
    print("="*60)
    print("LOCAL MODEL PROMPT OPTIMIZATION")
    print("="*60)

    # Create Ollama provider (no API key needed!)
    try:
        provider = OllamaProvider(
            model="llama2",  # or mistral, codellama, phi, etc.
            host="http://localhost:11434"
        )
        print("\n✓ Connected to Ollama")
        print(f"  Model: llama2")

        # List available models
        try:
            models = provider.list_models()
            print(f"\n  Available models: {', '.join(models)}")
        except Exception as e:
            print(f"  Could not list models: {e}")

    except ImportError:
        print("\n✗ Ollama package not installed")
        print("  Install with: pip install ollama")
        return
    except Exception as e:
        print(f"\n✗ Could not connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return

    # Test cases
    test_cases = [
        {"text": "This is great!", "expected": "positive"},
        {"text": "I don't like this", "expected": "negative"},
        {"text": "It's okay", "expected": "neutral"},
    ]

    # Create evaluator
    evaluator = SimpleTaskEvaluator()

    # Create optimizer
    # We use Hill Climbing since it's efficient for local models
    optimizer = HillClimbingOptimizer(
        provider=provider,
        evaluator=evaluator,
        num_neighbors=3,
        max_iterations=10,
        verbose=True
    )

    print("\nStarting optimization (this may take a few minutes with local models)...")

    # Initial prompt
    initial_prompt = "Classify: {text}"

    # Run optimization
    try:
        result = await optimizer.optimize(
            initial_prompt=initial_prompt,
            test_cases=test_cases
        )

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"\nBest Prompt: {result.best_prompt.content}")
        print(f"Score: {result.best_score:.2%}")
        print(f"Duration: {result.duration:.1f}s")

        print("\n✓ Successfully optimized prompt using local model!")
        print("  No API costs, complete privacy!")

    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
