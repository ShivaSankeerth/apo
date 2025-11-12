"""Example using MetaPromptOptimizer (Arize-inspired)."""

import asyncio
from apo import MetaPromptOptimizer, AnthropicProvider
from apo.core import Evaluator, EvaluationResult


class QuestionAnsweringEvaluator(Evaluator):
    """Evaluator for question answering tasks."""

    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    async def evaluate(self, prompt, test_cases):
        """Evaluate prompt on QA task."""
        correct = 0
        total = len(test_cases)

        for test_case in test_cases:
            # Render prompt with question
            full_prompt = f"{prompt.content}\n\nQuestion: {test_case['question']}\nAnswer:"

            # Get response
            response = await self.provider.complete(full_prompt, temperature=0)
            prediction = response.content.strip().lower()

            # Check correctness
            expected = test_case['answer'].lower()
            if expected in prediction or prediction in expected:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": accuracy},
            metadata={"correct": correct, "total": total}
        )


async def main():
    # Set up provider
    provider = AnthropicProvider(
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"
    )

    # Define test cases
    test_cases = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is 2 + 2?", "answer": "4"},
        {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
        {"question": "What is the largest planet?", "answer": "Jupiter"},
        {"question": "What is H2O?", "answer": "Water"},
    ]

    # Create evaluator
    evaluator = QuestionAnsweringEvaluator(provider)

    # Set up meta-prompt optimizer
    optimizer = MetaPromptOptimizer(
        provider=provider,
        evaluator=evaluator,
        max_iterations=10,
        patience=3,
        verbose=True
    )

    # Initial prompt
    initial_prompt = "Answer the following question concisely and accurately."

    # Run optimization
    print("Starting meta-prompt optimization...")
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
    print(f"Duration: {result.duration:.2f} seconds")
    print(f"\nInitial Prompt:\n{initial_prompt}")
    print(f"\nOptimized Prompt:\n{result.best_prompt.content}")


if __name__ == "__main__":
    asyncio.run(main())
