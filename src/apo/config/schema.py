"""Configuration schemas using Pydantic."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for LLM provider."""

    provider_type: str = Field(..., description="Type of provider (openai, anthropic)")
    api_key: str = Field(..., description="API key for the provider")
    model: str = Field(..., description="Model name to use")
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class OptimizerConfig(BaseModel):
    """Configuration for optimizer."""

    strategy: str = Field(..., description="Optimization strategy (genetic, hill_climbing, etc.)")
    max_iterations: int = Field(default=100, description="Maximum iterations")
    patience: int = Field(default=10, description="Early stopping patience")
    verbose: bool = Field(default=True, description="Verbose output")

    # Strategy-specific parameters
    population_size: Optional[int] = Field(default=None, description="Population size for genetic algorithm")
    generations: Optional[int] = Field(default=None, description="Generations for genetic algorithm")
    mutation_rate: Optional[float] = Field(default=None, description="Mutation rate for genetic algorithm")
    crossover_rate: Optional[float] = Field(default=None, description="Crossover rate for genetic algorithm")

    num_neighbors: Optional[int] = Field(default=None, description="Neighbors for hill climbing")

    initial_temperature: Optional[float] = Field(default=None, description="Initial temp for simulated annealing")
    cooling_rate: Optional[float] = Field(default=None, description="Cooling rate for simulated annealing")

    num_samples: Optional[int] = Field(default=None, description="Samples for random search")


class Config(BaseModel):
    """Main configuration."""

    provider: ProviderConfig
    optimizer: OptimizerConfig
    experiment_name: Optional[str] = Field(default=None, description="Name for the experiment")
    save_results: bool = Field(default=True, description="Whether to save results")
    results_dir: str = Field(default="experiments", description="Directory for results")

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
