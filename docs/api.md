# API Reference

## Core Classes

### Prompt

Represents a prompt with its content and metadata.

```python
class Prompt:
    content: str
    variables: Dict[str, Any]
    metadata: Dict[str, Any]
    score: Optional[float]

    def render(**kwargs) -> str
    def copy() -> Prompt
```

### PromptTemplate

Template for generating prompt variations.

```python
class PromptTemplate:
    base_template: str
    components: List[str]
    instructions: List[str]
    examples: List[str]

    def build(**kwargs) -> Prompt
    def add_instruction(instruction: str) -> None
    def add_example(example: str) -> None
```

### Evaluator

Abstract base class for prompt evaluators.

```python
class Evaluator(ABC):
    def __init__(metric_weights: Optional[Dict[str, float]] = None)

    @abstractmethod
    async def evaluate(prompt: Prompt, test_cases: List[Dict]) -> EvaluationResult

    async def evaluate_batch(prompts: List[Prompt], test_cases: List[Dict]) -> List[EvaluationResult]
```

### Optimizer

Abstract base class for optimizers.

```python
class Optimizer(ABC):
    def __init__(
        provider: LLMProvider,
        evaluator: Evaluator,
        max_iterations: int = 100,
        patience: int = 10,
        verbose: bool = True
    )

    @abstractmethod
    async def optimize(
        initial_prompt: str,
        test_cases: List[Dict],
        **kwargs
    ) -> OptimizationResult
```

## Optimization Strategies

### GeneticOptimizer

Genetic algorithm-based optimization.

```python
GeneticOptimizer(
    provider: LLMProvider,
    evaluator: Evaluator,
    population_size: int = 10,
    generations: int = 20,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.7,
    elite_size: int = 2,
    max_iterations: int = 100,
    patience: int = 10,
    verbose: bool = True
)
```

### HillClimbingOptimizer

Hill climbing optimization.

```python
HillClimbingOptimizer(
    provider: LLMProvider,
    evaluator: Evaluator,
    num_neighbors: int = 5,
    max_iterations: int = 50,
    patience: int = 10,
    verbose: bool = True
)
```

### SimulatedAnnealingOptimizer

Simulated annealing optimization.

```python
SimulatedAnnealingOptimizer(
    provider: LLMProvider,
    evaluator: Evaluator,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
    max_iterations: int = 100,
    patience: int = 20,
    verbose: bool = True
)
```

### RandomSearchOptimizer

Random search baseline.

```python
RandomSearchOptimizer(
    provider: LLMProvider,
    evaluator: Evaluator,
    num_samples: int = 50,
    max_iterations: int = 100,
    patience: int = 10,
    verbose: bool = True
)
```

## LLM Providers

### AnthropicProvider

```python
AnthropicProvider(
    api_key: str,
    model: str = "claude-3-5-sonnet-20241022",
    **kwargs
)

async def complete(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> LLMResponse
async def complete_batch(prompts: List[str], ...) -> List[LLMResponse]
async def generate_prompt_variations(base_prompt: str, num_variations: int = 5) -> List[str]
```

### OpenAIProvider

```python
OpenAIProvider(
    api_key: str,
    model: str = "gpt-4-turbo-preview",
    **kwargs
)

# Same methods as AnthropicProvider
```

## Experiment Tracking

### ExperimentTracker

```python
class ExperimentTracker:
    def __init__(experiment_dir: str = "experiments")

    def save_result(result: OptimizationResult, name: Optional[str] = None) -> Path
    def load_result(name: str) -> Dict[str, Any]
    def list_experiments() -> List[str]
    def compare_experiments(names: List[str]) -> Dict[str, Any]
    def get_summary_statistics(name: str) -> Dict[str, Any]
```

## Configuration

### Config

```python
class Config(BaseModel):
    provider: ProviderConfig
    optimizer: OptimizerConfig
    experiment_name: Optional[str]
    save_results: bool
    results_dir: str

    @classmethod
    def from_yaml(path: str) -> Config

    @classmethod
    def from_json(path: str) -> Config

    def to_yaml(path: str) -> None
    def to_json(path: str) -> None
```

## Metrics

Common metrics for evaluation:

```python
def accuracy_score(predictions: List[str], targets: List[str]) -> float
def semantic_similarity(text1: str, text2: str) -> float
def token_efficiency(prompt: str, max_tokens: int = 1000) -> float
def response_quality(response: str, criteria: Dict[str, Any]) -> float
```
