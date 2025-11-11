# Getting Started with APO

APO (Automatic Prompt Optimization) is a framework for systematically improving prompts for Large Language Models through various optimization algorithms.

## Installation

```bash
git clone https://github.com/ShivaSankeerth/apo.git
cd apo
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Optimization

```python
import asyncio
from apo import GeneticOptimizer, AnthropicProvider
from apo.core import Evaluator, EvaluationResult

# Create your evaluator
class MyEvaluator(Evaluator):
    async def evaluate(self, prompt, test_cases):
        # Your evaluation logic here
        score = 0.0
        # ... compute score based on prompt performance
        return EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": score}
        )

async def main():
    # Set up provider
    provider = AnthropicProvider(
        api_key="your-api-key",
        model="claude-3-5-sonnet-20241022"
    )

    # Create optimizer
    optimizer = GeneticOptimizer(
        provider=provider,
        evaluator=MyEvaluator(),
        population_size=10,
        generations=5
    )

    # Run optimization
    result = await optimizer.optimize(
        initial_prompt="Your initial prompt here",
        test_cases=[
            {"input": "example1", "expected": "output1"},
            # ... more test cases
        ]
    )

    print(f"Best prompt: {result.best_prompt.content}")
    print(f"Best score: {result.best_score}")

asyncio.run(main())
```

### 2. Choosing an Optimization Strategy

APO supports multiple optimization strategies:

- **GeneticOptimizer**: Uses evolutionary algorithms with crossover and mutation
- **HillClimbingOptimizer**: Iteratively improves by exploring neighbors
- **SimulatedAnnealingOptimizer**: Uses temperature-based acceptance for exploration
- **RandomSearchOptimizer**: Baseline that randomly samples variations

Example:

```python
from apo import HillClimbingOptimizer

optimizer = HillClimbingOptimizer(
    provider=provider,
    evaluator=evaluator,
    num_neighbors=5,
    max_iterations=20
)
```

### 3. Custom Evaluators

Create custom evaluators by subclassing `Evaluator`:

```python
from apo.core import Evaluator, EvaluationResult

class CustomEvaluator(Evaluator):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    async def evaluate(self, prompt, test_cases):
        total_score = 0
        for test_case in test_cases:
            # Render prompt with test data
            rendered = prompt.render(**test_case)

            # Get LLM response
            response = await self.provider.complete(rendered)

            # Score the response
            score = self.score_response(response, test_case)
            total_score += score

        avg_score = total_score / len(test_cases)

        return EvaluationResult(
            prompt=prompt,
            metrics={"custom_metric": avg_score}
        )

    def score_response(self, response, test_case):
        # Your scoring logic
        return 0.5
```

### 4. Multi-Objective Optimization

Optimize for multiple objectives simultaneously:

```python
from apo.core.evaluator import MultiObjectiveEvaluator

evaluator = MultiObjectiveEvaluator(
    evaluators=[
        AccuracyEvaluator(provider),
        EfficiencyEvaluator(),
        ClarityEvaluator()
    ],
    metric_weights={
        "accuracy": 0.6,
        "efficiency": 0.3,
        "clarity": 0.1
    }
)
```

### 5. Tracking Experiments

Save and compare optimization runs:

```python
from apo.tracking import ExperimentTracker

tracker = ExperimentTracker("my_experiments")

# Save result
tracker.save_result(result, name="experiment_1")

# Load and compare
data = tracker.load_result("experiment_1")
comparison = tracker.compare_experiments(["experiment_1", "experiment_2"])
```

### 6. Configuration Files

Use YAML/JSON configuration for reproducibility:

```yaml
# config.yaml
provider:
  provider_type: anthropic
  api_key: ${ANTHROPIC_API_KEY}
  model: claude-3-5-sonnet-20241022

optimizer:
  strategy: genetic
  max_iterations: 50
  patience: 10
  population_size: 20
  generations: 10

experiment_name: my_optimization
save_results: true
```

Load and use:

```python
from apo.config import Config

config = Config.from_yaml("config.yaml")
# Use config to set up optimizer...
```

## Next Steps

- Check out the [examples/](../examples) directory for complete examples
- Read the [API documentation](api.md) for detailed reference
- See [advanced usage](advanced.md) for more complex scenarios
