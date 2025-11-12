# APO: Automatic Prompt Optimization

A powerful framework for automatically optimizing prompts for Large Language Models (LLMs), combining the best ideas from Arize Phoenix and DSPy/GEPA.

## Features

### Classic Optimization Strategies
- **Genetic Algorithms** - Population-based evolution with crossover and mutation
- **Hill Climbing** - Iterative local search
- **Simulated Annealing** - Temperature-based stochastic optimization
- **Random Search** - Baseline random sampling

### Advanced Strategies (Inspired by Arize & DSPy)
- **Meta-Prompt Optimization** - LLM reflects on prompts and generates improvements (Arize-inspired)
- **Reflection-based Evolution** - LLM analyzes what works/doesn't work (GEPA-inspired)
- **Pareto Frontier Tracking** - Maintains multiple complementary strategies (GEPA-inspired)
- **Few-Shot Example Optimization** - Optimizes both template and examples
- **Bootstrap Learning** - Automatically generates examples from successful runs (DSPy-inspired)

### Core Capabilities
- **LLM Provider Support** - OpenAI, Anthropic (Claude), extensible to others
- **Flexible Evaluation System** - Custom metrics, multi-objective optimization
- **Prompt Signatures** - Declarative input/output specifications (DSPy-style)
- **Version Management** - Track and compare prompt versions
- **Experiment Tracking** - Complete run history and analysis

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Classic Genetic Algorithm
```python
from apo import GeneticOptimizer, AnthropicProvider
from apo.core import Evaluator, EvaluationResult

class SentimentEvaluator(Evaluator):
    async def evaluate(self, prompt, test_cases):
        # Your evaluation logic
        return EvaluationResult(prompt=prompt, metrics={"accuracy": 0.8})

provider = AnthropicProvider(api_key="your-api-key")
optimizer = GeneticOptimizer(provider=provider, evaluator=SentimentEvaluator())

result = await optimizer.optimize(
    initial_prompt="Classify sentiment: {text}",
    test_cases=test_cases
)
```

### Meta-Prompt Optimization (Arize-inspired)
```python
from apo import MetaPromptOptimizer

# LLM reflects on feedback and improves the prompt
optimizer = MetaPromptOptimizer(provider=provider, evaluator=evaluator)
result = await optimizer.optimize(initial_prompt, test_cases)
```

### Reflection with Pareto Frontier (GEPA-inspired)
```python
from apo import ReflectionOptimizer

# Maintains multiple complementary strategies
optimizer = ReflectionOptimizer(
    provider=provider,
    evaluator=evaluator,
    use_pareto_frontier=True
)
result = await optimizer.optimize(initial_prompt, test_cases)

# Access Pareto frontier
frontier = optimizer.get_pareto_frontier()
print(f"Frontier size: {frontier.size()}")
```

### Bootstrap Learning (DSPy-inspired)
```python
from apo import BootstrapOptimizer

# Automatically generates few-shot examples from successful predictions
optimizer = BootstrapOptimizer(provider=provider, evaluator=evaluator)
result = await optimizer.optimize(initial_prompt, test_cases)
print(f"Bootstrapped {result.metadata['num_examples']} examples")
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

## What Makes APO Different?

APO combines the best ideas from leading frameworks:

| Feature | APO | Arize Phoenix | DSPy/GEPA |
|---------|-----|---------------|-----------|
| Meta-prompt optimization | ✅ | ✅ | ❌ |
| Pareto frontier tracking | ✅ | ❌ | ✅ |
| Reflection-based evolution | ✅ | Partial | ✅ |
| Classic algorithms (GA, SA, HC) | ✅ | ❌ | ❌ |
| Bootstrap learning | ✅ | ❌ | ✅ |
| Prompt signatures | ✅ | ❌ | ✅ |
| Version management | ✅ | ✅ | ❌ |
| Experiment tracking | ✅ | ✅ | Partial |
| Pure optimization focus | ✅ | ❌ | ❌ |

See [docs/comparison.md](docs/comparison.md) for detailed comparison.

## Examples

- `examples/simple_optimization.py` - Basic genetic algorithm
- `examples/meta_prompt_example.py` - Meta-prompt optimization
- `examples/reflection_pareto_example.py` - Reflection with Pareto frontier
- `examples/bootstrap_example.py` - Bootstrap learning
- `examples/multi_objective.py` - Multi-objective optimization
- `examples/config_based.py` - Configuration-driven optimization

## Documentation

- [Getting Started](docs/getting_started.md) - Installation and basic usage
- [API Reference](docs/api.md) - Complete API documentation
- [Comparison](docs/comparison.md) - Comparison with other frameworks

## Citation

If you use APO in your research, please cite:

```bibtex
@software{apo2025,
  title={APO: Automatic Prompt Optimization Framework},
  author={APO Contributors},
  year={2025},
  url={https://github.com/ShivaSankeerth/apo}
}
```

## Acknowledgments

APO is inspired by:
- **Arize Phoenix** - Meta-prompt optimization and observability patterns
- **DSPy/GEPA** - Pareto frontier tracking and reflection-based evolution
- Academic research in prompt engineering and optimization

## License

MIT
