"""Tests for configuration schemas."""

import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError
from apo.config.schema import ProviderConfig, OptimizerConfig, Config


def test_provider_config_creation():
    """Test creating a provider config."""
    config = ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        model="gpt-4"
    )

    assert config.provider_type == "openai"
    assert config.api_key == "test-key"
    assert config.model == "gpt-4"
    assert config.additional_params == {}


def test_provider_config_with_additional_params():
    """Test provider config with additional parameters."""
    config = ProviderConfig(
        provider_type="anthropic",
        api_key="test-key",
        model="claude-3-opus",
        additional_params={"temperature": 0.7, "max_tokens": 1000}
    )

    assert config.additional_params["temperature"] == 0.7
    assert config.additional_params["max_tokens"] == 1000


def test_provider_config_validation():
    """Test provider config validation."""
    # Missing required fields should raise error
    with pytest.raises(ValidationError):
        ProviderConfig(provider_type="openai")


def test_optimizer_config_creation():
    """Test creating an optimizer config."""
    config = OptimizerConfig(
        strategy="genetic",
        max_iterations=100,
        patience=10
    )

    assert config.strategy == "genetic"
    assert config.max_iterations == 100
    assert config.patience == 10


def test_optimizer_config_defaults():
    """Test optimizer config default values."""
    config = OptimizerConfig(strategy="hill_climbing")

    assert config.max_iterations == 100
    assert config.patience == 10
    assert config.verbose is True


def test_optimizer_config_genetic_params():
    """Test optimizer config with genetic algorithm parameters."""
    config = OptimizerConfig(
        strategy="genetic",
        population_size=20,
        generations=50,
        mutation_rate=0.2,
        crossover_rate=0.7
    )

    assert config.population_size == 20
    assert config.generations == 50
    assert config.mutation_rate == 0.2
    assert config.crossover_rate == 0.7


def test_optimizer_config_hill_climbing_params():
    """Test optimizer config with hill climbing parameters."""
    config = OptimizerConfig(
        strategy="hill_climbing",
        num_neighbors=5
    )

    assert config.num_neighbors == 5


def test_optimizer_config_simulated_annealing_params():
    """Test optimizer config with simulated annealing parameters."""
    config = OptimizerConfig(
        strategy="simulated_annealing",
        initial_temperature=100.0,
        cooling_rate=0.95
    )

    assert config.initial_temperature == 100.0
    assert config.cooling_rate == 0.95


def test_main_config_creation():
    """Test creating main config."""
    provider_config = ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        model="gpt-4"
    )
    optimizer_config = OptimizerConfig(strategy="genetic")

    config = Config(
        provider=provider_config,
        optimizer=optimizer_config
    )

    assert config.provider.provider_type == "openai"
    assert config.optimizer.strategy == "genetic"


def test_main_config_defaults():
    """Test main config default values."""
    provider_config = ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        model="gpt-4"
    )
    optimizer_config = OptimizerConfig(strategy="genetic")

    config = Config(
        provider=provider_config,
        optimizer=optimizer_config
    )

    assert config.experiment_name is None
    assert config.save_results is True
    assert config.results_dir == "experiments"


def test_config_from_yaml():
    """Test loading config from YAML file."""
    yaml_content = """
provider:
  provider_type: anthropic
  api_key: test-key
  model: claude-3-opus
  additional_params:
    temperature: 0.7

optimizer:
  strategy: genetic
  max_iterations: 50
  population_size: 10

experiment_name: test_experiment
save_results: true
results_dir: my_experiments
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    try:
        config = Config.from_yaml(yaml_path)

        assert config.provider.provider_type == "anthropic"
        assert config.provider.model == "claude-3-opus"
        assert config.optimizer.strategy == "genetic"
        assert config.optimizer.max_iterations == 50
        assert config.experiment_name == "test_experiment"
    finally:
        Path(yaml_path).unlink()


def test_config_to_yaml():
    """Test saving config to YAML file."""
    provider_config = ProviderConfig(
        provider_type="openai",
        api_key="test-key",
        model="gpt-4"
    )
    optimizer_config = OptimizerConfig(
        strategy="genetic",
        max_iterations=100
    )
    config = Config(
        provider=provider_config,
        optimizer=optimizer_config,
        experiment_name="test"
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name

    try:
        config.to_yaml(yaml_path)

        # Load it back and verify
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        assert data["provider"]["provider_type"] == "openai"
        assert data["optimizer"]["strategy"] == "genetic"
        assert data["experiment_name"] == "test"
    finally:
        Path(yaml_path).unlink()


def test_config_validation():
    """Test config validation."""
    # Missing provider should raise error
    with pytest.raises(ValidationError):
        Config(optimizer=OptimizerConfig(strategy="genetic"))

    # Missing optimizer should raise error
    with pytest.raises(ValidationError):
        Config(provider=ProviderConfig(
            provider_type="openai",
            api_key="key",
            model="gpt-4"
        ))
