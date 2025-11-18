"""Example using Prompt Diff Viewer.

This example shows how to visualize changes between prompts
as they evolve through optimization.
"""

import asyncio
from apo import MetaPromptOptimizer, AnthropicProvider
from apo.core import Evaluator, EvaluationResult
from apo.utils import PromptDiff, show_diff, PromptEvolutionTracker


class SimpleEvaluator(Evaluator):
    """Simple evaluator for demonstration."""

    async def evaluate(self, prompt, test_cases):
        return EvaluationResult(
            prompt=prompt,
            metrics={"score": 0.75}  # Placeholder
        )


async def optimization_with_diff_tracking():
    """Run optimization and track prompt evolution."""

    # This would use real providers in practice
    print("="*60)
    print("PROMPT EVOLUTION WITH DIFF TRACKING")
    print("="*60)

    # Simulate prompt evolution
    prompts = [
        ("Initial", "Classify the sentiment: {text}"),
        ("Iteration 1", "Classify the sentiment of the following text as positive, negative, or neutral: {text}"),
        ("Iteration 2", "Analyze the sentiment of the following text.\n\nText: {text}\n\nClassify as: positive, negative, or neutral\n\nSentiment:"),
        ("Final", "Analyze the emotional tone and sentiment of the following text. Consider context, word choice, and overall message.\n\nText: {text}\n\nProvide a clear classification:\n- positive\n- negative  \n- neutral\n\nSentiment:")
    ]

    # Track evolution
    tracker = PromptEvolutionTracker()
    for name, prompt in prompts:
        tracker.add_version(name, prompt)

    # Show evolution summary
    tracker.show_evolution(show_all_diffs=False)

    # Show detailed diff between first and last
    print("\n" + "="*60)
    print("DETAILED DIFF: Initial → Final")
    print("="*60)

    diff = PromptDiff(prompts[0][1], prompts[-1][1])

    # Console view
    print("\nConsole View:")
    print(diff.to_console(colors=True))

    # Statistics
    print("\nStatistics:")
    stats = diff.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\nSimilarity: {diff.get_similarity():.1%}")

    # Save HTML version
    html_output = diff.to_html()
    with open("prompt_diff.html", "w") as f:
        f.write(html_output)
    print("\n✓ Saved HTML diff to: prompt_diff.html")

    # Markdown version
    print("\nMarkdown View:")
    print(diff.to_markdown())

    # Find biggest change
    best_improvement = tracker.get_best_improvement()
    if best_improvement:
        from_ver, to_ver, change_pct = best_improvement
        print(f"\nBiggest single improvement:")
        print(f"  {from_ver} → {to_ver}")
        print(f"  Change: {change_pct:.1f}%")


def simple_diff_example():
    """Simple example of comparing two prompts."""

    print("\n" + "="*60)
    print("SIMPLE DIFF EXAMPLE")
    print("="*60)

    original = "Answer the question: {question}"

    modified = """Answer the following question concisely and accurately.

Question: {question}

Provide a clear, direct answer:"""

    # Show diff
    print("\nChanges:")
    print(show_diff(original, modified, format="console"))

    # Get stats
    diff = PromptDiff(original, modified)
    stats = diff.get_stats()

    print(f"\nSummary:")
    print(f"  Added {stats['added_lines']} lines")
    print(f"  Removed {stats['removed_lines']} lines")
    print(f"  Net change: {stats['net_change_chars']:+d} characters")
    print(f"  Similarity: {diff.get_similarity():.1%}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(optimization_with_diff_tracking())

    print("\n\n")

    simple_diff_example()
