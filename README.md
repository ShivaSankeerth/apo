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
- **LLM Provider Support** - OpenAI, Anthropic (Claude), Google Gemini, Ollama (local models), Cohere, Mistral
- **LLM-as-Judge Evaluation** - Use powerful LLMs to evaluate prompt quality with customizable rubrics
- **Prompt Diff Viewer** - Visualize changes between prompts with console, HTML, and markdown outputs
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

### Optional Provider Dependencies

Install specific providers as needed:

```bash
# Google Gemini
pip install -e ".[gemini]"

# Ollama (local models)
pip install -e ".[ollama]"

# Cohere
pip install -e ".[cohere]"

# Mistral AI
pip install -e ".[mistral]"

# All providers at once
pip install -e ".[all-providers]"
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

### LLM-as-Judge Evaluation
```python
from apo.evaluators import LLMJudgeEvaluator, ACCURACY_RUBRIC, HELPFULNESS_RUBRIC

# Use a powerful LLM to judge prompt outputs
judge_provider = AnthropicProvider(api_key="your-key", model="claude-3-5-sonnet-20241022")
evaluator = LLMJudgeEvaluator(
    judge_provider=judge_provider,
    rubrics=[ACCURACY_RUBRIC, HELPFULNESS_RUBRIC],
    use_chain_of_thought=True
)

optimizer = MetaPromptOptimizer(provider=provider, evaluator=evaluator)
result = await optimizer.optimize(initial_prompt, test_cases)
```

### Local Models with Ollama
```python
from apo.providers import OllamaProvider

# No API costs - run optimization completely locally!
provider = OllamaProvider(model="llama2", host="http://localhost:11434")
optimizer = HillClimbingOptimizer(provider=provider, evaluator=evaluator)
result = await optimizer.optimize(initial_prompt, test_cases)
```

### Prompt Diff Viewer
```python
from apo.utils import PromptDiff, PromptEvolutionTracker

# Compare two prompts
diff = PromptDiff(original_prompt, optimized_prompt)
print(diff.to_console(colors=True))
print(f"Similarity: {diff.get_similarity():.1%}")

# Track evolution over time
tracker = PromptEvolutionTracker()
tracker.add_version("Initial", prompt_v1)
tracker.add_version("Iteration 1", prompt_v2)
tracker.add_version("Final", prompt_v3)
tracker.show_evolution()

# Save HTML diff
with open("diff.html", "w") as f:
    f.write(diff.to_html())
```

## Architecture

```
src/apo/
├── core/           # Core abstractions (Prompt, Optimizer, Evaluator)
├── strategies/     # Optimization algorithms
├── providers/      # LLM provider integrations (OpenAI, Anthropic, Gemini, Ollama, Cohere, Mistral)
├── evaluators/     # Advanced evaluators (LLM-as-Judge)
├── utils/          # Utilities (Prompt Diff Viewer)
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

## Interactive App

Try the Streamlit web application for visual prompt optimization:

```bash
pip install -r requirements-app.txt
streamlit run app.py
```

Features:
- 📝 Interactive prompt editor with variable detection
- 🧪 Dynamic test case management
- ▶️ Live testing with LLM outputs
- 👍👎 User feedback collection
- 🚀 AI-powered optimization with progress visualization
- 📊 Before/after comparison and metrics

[Read the App Guide](APP_README.md)

## Benchmark Suite

Compare all optimizers with comprehensive benchmarks:

```bash
# Full benchmark (4 tasks, 8 optimizers)
python run_benchmark.py --provider anthropic --api-key YOUR_KEY

# Quick test (1 task)
python run_benchmark.py --provider anthropic --api-key YOUR_KEY --quick
```

**Results include:**
- 📊 Interactive performance charts
- 📈 Convergence curves
- ⚡ Efficiency analysis
- 🏆 Winner rankings
- 📑 Detailed metrics tables

**Tested optimizers:** Random Search, Hill Climbing, Simulated Annealing, Genetic Algorithm, Meta-Prompt, Reflection, Reflection+Pareto, Bootstrap

**Benchmark tasks:** Sentiment Classification, Question Answering, Category Classification, Instruction Following

[Read the Benchmark Guide](benchmarks/BENCHMARK_README.md)

## Examples

### Optimization Strategies
- `examples/simple_optimization.py` - Basic genetic algorithm
- `examples/meta_prompt_example.py` - Meta-prompt optimization
- `examples/reflection_pareto_example.py` - Reflection with Pareto frontier
- `examples/bootstrap_example.py` - Bootstrap learning
- `examples/multi_objective.py` - Multi-objective optimization
- `examples/config_based.py` - Configuration-driven optimization

### Advanced Features
- `examples/llm_judge_example.py` - LLM-as-Judge evaluation with custom rubrics
- `examples/local_model_example.py` - Local optimization using Ollama (no API costs!)
- `examples/diff_viewer_example.py` - Visualizing prompt evolution and changes

## Documentation

- [Getting Started](docs/getting_started.md) - Installation and basic usage
- [API Reference](docs/api.md) - Complete API documentation
- [Comparison](docs/comparison.md) - Comparison with other frameworks
- [Streamlit App Guide](docs/streamlit_app_guide.md) - Interactive app walkthrough
- [Benchmark Guide](benchmarks/BENCHMARK_README.md) - Running and analyzing benchmarks

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
