# APO Benchmark Suite

Comprehensive benchmarking system for comparing all APO optimizers across multiple tasks.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-benchmark.txt
```

### 2. Run Benchmark

**Full Benchmark** (all tasks, ~30-60 minutes):
```bash
python run_benchmark.py --provider anthropic --api-key YOUR_API_KEY
```

**Quick Benchmark** (sentiment task only, ~5-10 minutes):
```bash
python run_benchmark.py --provider anthropic --api-key YOUR_API_KEY --quick
```

**Using OpenAI**:
```bash
python run_benchmark.py --provider openai --api-key YOUR_API_KEY
```

### 3. View Results

Results are saved to `benchmark_results/` directory:
- `results.json` - Raw benchmark data
- `summary.csv` - Summary table
- `performance_comparison.html` - Interactive performance charts
- `convergence_*.html` - Convergence curves for each task
- `efficiency_analysis.html` - Time and evaluation efficiency
- `winners.html` - Best optimizer by task

Open any `.html` file in your browser to view interactive visualizations.

## What Gets Benchmarked

### Optimizers Tested

1. **Random Search** - Baseline random sampling
2. **Hill Climbing** - Local search optimization
3. **Simulated Annealing** - Temperature-based stochastic search
4. **Genetic Algorithm** - Population-based evolution
5. **Meta-Prompt** - LLM reflection optimizer (Arize-inspired)
6. **Reflection** - LLM-based analysis (GEPA-inspired)
7. **Reflection+Pareto** - With Pareto frontier tracking
8. **Bootstrap** - Automatic example generation (DSPy-inspired)

### Tasks

1. **Sentiment Classification** (10 test cases)
   - Classify text as positive/negative/neutral
   - Initial prompt: "Classify the sentiment: {text}"

2. **Question Answering** (10 test cases)
   - Answer factual questions
   - Initial prompt: "Answer this question: {question}"

3. **Category Classification** (10 test cases)
   - Categorize news headlines
   - Initial prompt: "Categorize this: {text}"

4. **Instruction Following** (10 test cases)
   - Follow simple instructions
   - Initial prompt: "Follow these instructions: {instruction}"

### Metrics Collected

For each optimizer × task combination:
- **Initial Score**: Performance before optimization
- **Final Score**: Performance after optimization
- **Improvement**: Score increase
- **Duration**: Time taken (seconds)
- **Evaluations**: Number of prompts evaluated
- **Iterations**: Optimization iterations
- **Convergence**: Iteration where best score was found
- **Score History**: Performance over time
- **Best Prompt**: Final optimized prompt

## Visualizations

### 1. Performance Comparison
Bar charts showing initial vs final scores for each optimizer on each task.

**Insights:**
- Which optimizers improve scores the most
- How much improvement is possible per task
- Consistency across tasks

### 2. Convergence Curves
Line plots showing score progression over iterations.

**Insights:**
- How quickly each optimizer converges
- Whether optimizers get stuck in local optima
- Sample efficiency

### 3. Efficiency Analysis
Scatter plots of improvement vs time/evaluations.

**Insights:**
- Which optimizers are fastest
- Which are most sample-efficient
- Trade-offs between speed and quality

### 4. Winner Chart
Bar chart showing the best optimizer for each task.

**Insights:**
- Overall best performers
- Task-specific strengths
- Consistency vs specialization

## Example Output

```
================================================================================
APO Optimizer Benchmark Suite
Starting at: 2025-01-18 10:30:00
Tasks: 4
================================================================================

============================================================
Benchmarking: Sentiment Classification
============================================================
[Benchmark] Running Random Search on Sentiment Classification...
  ✓ Random Search: 60.0% → 75.0% (+15.0%) in 12.3s
[Benchmark] Running Hill Climbing on Sentiment Classification...
  ✓ Hill Climbing: 60.0% → 82.0% (+22.0%) in 15.7s
[Benchmark] Running Meta-Prompt on Sentiment Classification...
  ✓ Meta-Prompt: 60.0% → 90.0% (+30.0%) in 8.4s
...

================================================================================
BENCHMARK SUMMARY
================================================================================
Optimizer           Task                    Initial  Final    Improvement  Time(s)
Random Search       Sentiment               60.0%    75.0%    +15.0%       12.3
Hill Climbing       Sentiment               60.0%    82.0%    +22.0%       15.7
Meta-Prompt         Sentiment               60.0%    90.0%    +30.0%       8.4
...

================================================================================
BEST OPTIMIZER BY TASK
================================================================================

Sentiment Classification:
  Winner: Meta-Prompt
  Score: 60.0% → 90.0% (+30.0%)
  Time: 8.4s

Question Answering:
  Winner: Reflection+Pareto
  Score: 70.0% → 92.0% (+22.0%)
  Time: 18.2s
...
```

## Customizing the Benchmark

### Add a New Task

Edit `benchmarks/tasks.py`:

```python
MY_TASK = BenchmarkTask(
    name="My Custom Task",
    initial_prompt="Your initial prompt: {input}",
    test_cases=[
        {"input": "example 1", "expected": "output 1"},
        {"input": "example 2", "expected": "output 2"},
        # Add more test cases
    ]
)
```

Add to `ALL_TASKS` list.

### Adjust Optimizer Parameters

Edit `benchmarks/runner.py` in the `run_all_optimizers` method:

```python
optimizers = [
    (MetaPromptOptimizer, "Meta-Prompt", {
        "max_iterations": 15,  # Increase for more optimization
        "patience": 5,         # Adjust early stopping
    }),
    # ...
]
```

### Custom Evaluation

Create a custom evaluator in `benchmarks/runner.py`:

```python
class MyCustomEvaluator(Evaluator):
    async def evaluate(self, prompt, test_cases):
        # Your custom evaluation logic
        return EvaluationResult(prompt=prompt, metrics={"my_metric": score})
```

## Interpreting Results

### High Initial Score (>70%)
- Task is easy or initial prompt is already good
- Less room for improvement
- Tests optimizer refinement capabilities

### Low Initial Score (<50%)
- Challenging task or poor initial prompt
- More room for improvement
- Tests optimizer's ability to find good solutions

### Fast Convergence (<5 iterations)
- Optimizer finds solution quickly
- May indicate greedy behavior
- Could miss global optimum

### Slow Convergence (>15 iterations)
- Thorough exploration
- More likely to find global optimum
- Higher API costs

### High Variance Across Tasks
- Optimizer has task-specific strengths
- May not generalize well
- Consider ensemble approaches

### Consistent Performance
- Robust optimizer
- Good general-purpose choice
- Reliable across different domains

## Tips for Better Benchmarks

1. **Use Multiple Test Cases** (10-20 per task)
   - More reliable scores
   - Better statistical significance

2. **Diverse Tasks**
   - Tests different aspects
   - Reveals optimizer strengths/weaknesses

3. **Multiple Runs**
   - Account for randomness
   - Calculate confidence intervals

4. **Control for Costs**
   - Track token usage
   - Compare cost-effectiveness
   - Consider timeout limits

5. **Baseline Comparison**
   - Always include random search
   - Shows minimum expected improvement
   - Validates evaluation

## Troubleshooting

### API Rate Limits
- Add delays between optimizer runs
- Use `--quick` flag for testing
- Run during off-peak hours

### Out of Memory
- Reduce number of test cases
- Lower population sizes
- Run tasks sequentially

### Inconsistent Results
- Check API provider status
- Increase test case diversity
- Run multiple times and average

### Slow Performance
- Use faster models (claude-sonnet, gpt-3.5)
- Reduce max_iterations
- Lower population/generation sizes

## Advanced Usage

### Parallel Execution

Run multiple tasks in parallel:

```python
import asyncio
from benchmarks.runner import BenchmarkRunner

async def parallel_benchmark():
    runner = BenchmarkRunner(provider)

    # Run tasks concurrently
    tasks = [
        runner.run_all_optimizers(task)
        for task in ALL_TASKS
    ]

    results = await asyncio.gather(*tasks)
```

### Custom Metrics

Track additional metrics:

```python
class DetailedEvaluator(Evaluator):
    async def evaluate(self, prompt, test_cases):
        # Custom metrics
        return EvaluationResult(
            prompt=prompt,
            metrics={
                "accuracy": accuracy,
                "latency": avg_latency,
                "token_efficiency": tokens_per_case,
                "robustness": score_variance
            }
        )
```

### Statistical Analysis

```python
import scipy.stats as stats

# Compare two optimizers
optimizer1_scores = [r.final_score for r in results if r.optimizer_name == "Meta-Prompt"]
optimizer2_scores = [r.final_score for r in results if r.optimizer_name == "Reflection"]

t_stat, p_value = stats.ttest_ind(optimizer1_scores, optimizer2_scores)
print(f"p-value: {p_value}")
```

## Contributing

To add new visualizations or analysis:

1. Add functions to `benchmarks/visualize.py`
2. Update `generate_all_visualizations()`
3. Add to this README

## Citation

When publishing benchmarks, cite:

```bibtex
@software{apo_benchmark2025,
  title={APO Optimizer Benchmark Suite},
  author={APO Contributors},
  year={2025},
  url={https://github.com/ShivaSankeerth/apo}
}
```
