"""Run comprehensive benchmark of all optimizers.

Usage:
    python run_benchmark.py --provider anthropic --api-key YOUR_KEY
    python run_benchmark.py --provider openai --api-key YOUR_KEY
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from apo import AnthropicProvider, OpenAIProvider
from benchmarks.runner import BenchmarkRunner
from benchmarks.tasks import ALL_TASKS, SENTIMENT_TASK
from benchmarks.visualize import generate_all_visualizations, create_summary_table


async def main():
    parser = argparse.ArgumentParser(description='Run APO optimizer benchmarks')
    parser.add_argument('--provider', choices=['anthropic', 'openai'], required=True,
                        help='LLM provider to use')
    parser.add_argument('--api-key', required=True, help='API key for the provider')
    parser.add_argument('--model', help='Specific model to use (optional)')
    parser.add_argument('--output-dir', default='benchmark_results',
                        help='Directory to save results')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark (only sentiment task)')

    args = parser.parse_args()

    # Create provider
    if args.provider == 'anthropic':
        model = args.model or 'claude-3-5-sonnet-20241022'
        provider = AnthropicProvider(api_key=args.api_key, model=model)
        print(f"Using Anthropic Claude: {model}")
    else:
        model = args.model or 'gpt-4-turbo-preview'
        provider = OpenAIProvider(api_key=args.api_key, model=model)
        print(f"Using OpenAI GPT: {model}")

    # Select tasks
    tasks = [SENTIMENT_TASK] if args.quick else ALL_TASKS
    print(f"Running benchmark on {len(tasks)} task(s)")

    # Run benchmark
    runner = BenchmarkRunner(provider, verbose=True)
    results = await runner.run_benchmark_suite(tasks)

    # Save raw results
    runner.save_results(f"{args.output_dir}/results.json")

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_all_visualizations(results, args.output_dir)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))

    # Print winners
    print("\n" + "="*80)
    print("BEST OPTIMIZER BY TASK")
    print("="*80)
    tasks_dict = {}
    for result in results:
        if result.task_name not in tasks_dict:
            tasks_dict[result.task_name] = []
        tasks_dict[result.task_name].append(result)

    for task_name, task_results in tasks_dict.items():
        best = max(task_results, key=lambda x: x.final_score)
        print(f"\n{task_name}:")
        print(f"  Winner: {best.optimizer_name}")
        print(f"  Score: {best.initial_score:.1%} → {best.final_score:.1%} (+{best.improvement:.1%})")
        print(f"  Time: {best.duration_seconds:.1f}s")

    print("\n" + "="*80)
    print(f"All results and visualizations saved to: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
