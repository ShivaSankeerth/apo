# APO: Automatic Prompt Optimization

A powerful framework for automatically optimizing prompts for Large Language Models (LLMs).

## Features

- **Multiple Optimization Strategies**
  - Genetic Algorithms
  - Hill Climbing
  - Simulated Annealing
  - Random Search
  - Bayesian Optimization

- **LLM Provider Support**
  - OpenAI (GPT-3.5, GPT-4, etc.)
  - Anthropic (Claude)
  - Extensible to other providers

- **Flexible Evaluation System**
  - Custom metrics
  - Multi-objective optimization
  - Built-in common metrics

- **Experiment Tracking**
  - Run history
  - Performance metrics
  - Best prompt tracking

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from apo import GeneticOptimizer, AnthropicProvider
from apo.core import Prompt, Evaluator

# Define your task
task = "Classify the sentiment of this text: {text}"

# Create a simple evaluator
class SentimentEvaluator(Evaluator):
    async def evaluate(self, prompt, test_cases):
        # Your evaluation logic here
        score = 0.0
        # ... evaluate the prompt
        return {"accuracy": score}

# Set up the optimizer
provider = AnthropicProvider(api_key="your-api-key")
evaluator = SentimentEvaluator()

optimizer = GeneticOptimizer(
    provider=provider,
    evaluator=evaluator,
    population_size=10,
    generations=5
)

# Run optimization
best_prompt = await optimizer.optimize(
    initial_prompt=task,
    test_cases=your_test_cases
)

print(f"Best prompt: {best_prompt}")
```

## Architecture

```
src/apo/
├── core/           # Core abstractions (Prompt, Optimizer, Evaluator)
├── strategies/     # Optimization algorithms
├── providers/      # LLM provider integrations
├── metrics/        # Evaluation metrics
├── tracking/       # Experiment tracking
└── config/         # Configuration management
```

## Documentation

See the `examples/` directory for detailed usage examples.

## License

MIT
