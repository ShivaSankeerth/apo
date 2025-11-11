"""Example using configuration files."""

import asyncio
from apo.config import Config
from apo import AnthropicProvider, GeneticOptimizer
from apo.core import Evaluator, EvaluationResult
from apo.tracking import ExperimentTracker


class SimpleEvaluator(Evaluator):
    """Simple placeholder evaluator."""

    async def evaluate(self, prompt, test_cases):
        # Simplified evaluation - in practice, use LLM here
        score = 0.8  # Placeholder
        return EvaluationResult(
            prompt=prompt,
            metrics={"score": score}
        )


async def main():
    # Load configuration from YAML
    # Create a sample config first
    config = Config(
        provider={
            "provider_type": "anthropic",
            "api_key": "your-api-key",
            "model": "claude-3-5-sonnet-20241022"
        },
        optimizer={
            "strategy": "genetic",
            "max_iterations": 20,
            "patience": 5,
            "population_size": 10,
            "generations": 5
        },
        experiment_name="sentiment_optimization",
        save_results=True
    )

    # Save config example
    config.to_yaml("examples/config.yaml")

    # Create provider based on config
    if config.provider.provider_type == "anthropic":
        provider = AnthropicProvider(
            api_key=config.provider.api_key,
            model=config.provider.model
        )
    else:
        raise ValueError(f"Unknown provider: {config.provider.provider_type}")

    # Create optimizer based on config
    optimizer = GeneticOptimizer(
        provider=provider,
        evaluator=SimpleEvaluator(),
        population_size=config.optimizer.population_size,
        generations=config.optimizer.generations,
        max_iterations=config.optimizer.max_iterations,
        patience=config.optimizer.patience,
        verbose=config.optimizer.verbose
    )

    # Run optimization
    result = await optimizer.optimize(
        initial_prompt="Classify sentiment",
        test_cases=[]
    )

    # Save results
    if config.save_results:
        tracker = ExperimentTracker(config.results_dir)
        saved_path = tracker.save_result(result, name=config.experiment_name)
        print(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    asyncio.run(main())
