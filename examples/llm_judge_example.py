"""Example using LLM-as-Judge for more nuanced evaluation.

This example shows how to use an LLM to evaluate prompt outputs
instead of simple string matching.
"""

import asyncio
from apo import MetaPromptOptimizer, AnthropicProvider
from apo.evaluators import LLMJudgeEvaluator, ACCURACY_RUBRIC, HELPFULNESS_RUBRIC


async def main():
    # Set up two providers:
    # 1. Provider for generating responses (the prompt we're optimizing)
    # 2. Provider for judging quality (usually a more powerful model)

    response_provider = AnthropicProvider(
        api_key="your-api-key-here",
        model="claude-3-sonnet-20240229"  # Faster, cheaper
    )

    judge_provider = AnthropicProvider(
        api_key="your-api-key-here",
        model="claude-3-5-sonnet-20241022"  # More capable judge
    )

    # Create LLM judge evaluator with custom rubrics
    evaluator = LLMJudgeEvaluator(
        judge_provider=judge_provider,
        rubrics=[ACCURACY_RUBRIC, HELPFULNESS_RUBRIC],
        use_chain_of_thought=True,  # Judge explains reasoning
        include_reasoning=True  # Include in results
    )

    # Test cases for customer support responses
    test_cases = [
        {
            "question": "How do I reset my password?",
            "expected": "Clear step-by-step instructions with a link to the reset page"
        },
        {
            "question": "When will my order arrive?",
            "expected": "Information about checking order status and typical delivery times"
        },
        {
            "question": "Can I return this product?",
            "expected": "Return policy explanation with timeframe and process"
        },
        {
            "question": "The product is broken, what should I do?",
            "expected": "Empathetic response with replacement or refund options"
        },
        {
            "question": "Do you ship internationally?",
            "expected": "Clear yes/no with list of supported countries or regions"
        },
    ]

    # Initial prompt (pretty basic)
    initial_prompt = "Answer the customer question: {question}"

    # Set up optimizer
    optimizer = MetaPromptOptimizer(
        provider=response_provider,
        evaluator=evaluator,
        max_iterations=5,
        patience=2,
        verbose=True
    )

    print("Starting optimization with LLM-as-Judge evaluation...")
    print("The judge will evaluate responses on Accuracy and Helpfulness.\n")

    # Run optimization
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        test_cases=test_cases
    )

    # Print results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)

    print(f"\nInitial Prompt:")
    print(f"  {initial_prompt}")

    print(f"\nOptimized Prompt:")
    print(f"  {result.best_prompt.content}")

    print(f"\nScores:")
    for eval_result in [result.history[0], result.history[-1]]:
        prompt_label = "Initial" if eval_result == result.history[0] else "Final"
        print(f"\n  {prompt_label}:")
        for metric, score in eval_result.metrics.items():
            print(f"    {metric}: {score:.2%}")

    # Show some reasoning from the judge
    if result.history[-1].metadata.get("reasoning"):
        print(f"\nJudge's Reasoning (sample):")
        reasoning_samples = result.history[-1].metadata["reasoning"][:1]
        for i, reasoning in enumerate(reasoning_samples, 1):
            print(f"\n  Example {i}:")
            print(f"  {reasoning[:200]}...")

    print(f"\nImprovement: {result.best_score - result.history[0].score:.2%}")
    print(f"Total Evaluations: {result.total_evaluations}")


if __name__ == "__main__":
    asyncio.run(main())
