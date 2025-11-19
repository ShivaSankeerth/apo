"""Tests for experiment tracking."""

import pytest
import tempfile
import json
from pathlib import Path
from apo.tracking.experiment import ExperimentTracker
from apo.core.optimizer import OptimizationResult
from apo.core.prompt import Prompt


@pytest.fixture
def temp_experiment_dir():
    """Create a temporary directory for experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_result():
    """Create a sample optimization result."""
    prompt = Prompt(content="Optimized prompt", score=0.9)
    return OptimizationResult(
        best_prompt=prompt,
        best_score=0.9,
        history=[prompt],
        iterations=10,
        metadata={"test": True}
    )


def test_experiment_tracker_initialization(temp_experiment_dir):
    """Test experiment tracker initialization."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    assert tracker.experiment_dir == Path(temp_experiment_dir)
    assert tracker.experiment_dir.exists()


def test_save_result(temp_experiment_dir, sample_result):
    """Test saving an optimization result."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    saved_path = tracker.save_result(sample_result, name="test_experiment")

    assert saved_path.exists()
    assert saved_path.suffix == ".json"
    assert "test_experiment" in saved_path.name


def test_save_result_content(temp_experiment_dir, sample_result):
    """Test that saved result contains correct data."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    saved_path = tracker.save_result(sample_result, name="test_exp")

    # Read and verify content
    with open(saved_path, 'r') as f:
        data = json.load(f)

    assert data["name"] == "test_exp"
    assert data["best_score"] == 0.9
    assert data["best_prompt"] == "Optimized prompt"
    assert "timestamp" in data


def test_save_result_auto_naming(temp_experiment_dir, sample_result):
    """Test automatic naming when no name provided."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    saved_path = tracker.save_result(sample_result)

    assert saved_path.exists()
    assert "experiment_" in saved_path.name


def test_save_result_with_metadata(temp_experiment_dir, sample_result):
    """Test saving result with additional metadata."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    metadata = {"custom_field": "custom_value", "run_id": 123}
    saved_path = tracker.save_result(sample_result, metadata=metadata)

    with open(saved_path, 'r') as f:
        data = json.load(f)

    assert data["metadata"]["custom_field"] == "custom_value"
    assert data["metadata"]["run_id"] == 123


def test_load_result(temp_experiment_dir, sample_result):
    """Test loading a saved result."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    # Save first
    saved_path = tracker.save_result(sample_result, name="load_test")

    # Load
    loaded_data = tracker.load_result("load_test")

    assert loaded_data["best_score"] == 0.9
    assert loaded_data["best_prompt"] == "Optimized prompt"


def test_list_experiments(temp_experiment_dir, sample_result):
    """Test listing all experiments."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    # Save multiple experiments
    tracker.save_result(sample_result, name="exp1")
    tracker.save_result(sample_result, name="exp2")
    tracker.save_result(sample_result, name="exp3")

    experiments = tracker.list_experiments()

    assert len(experiments) >= 3
    names = [exp["name"] for exp in experiments]
    assert "exp1" in names
    assert "exp2" in names
    assert "exp3" in names


def test_compare_experiments(temp_experiment_dir):
    """Test comparing multiple experiments."""
    tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

    # Create results with different scores
    result1 = OptimizationResult(
        best_prompt=Prompt(content="Prompt 1", score=0.7),
        best_score=0.7,
        history=[],
        iterations=5,
        metadata={}
    )
    result2 = OptimizationResult(
        best_prompt=Prompt(content="Prompt 2", score=0.9),
        best_score=0.9,
        history=[],
        iterations=8,
        metadata={}
    )

    tracker.save_result(result1, name="exp_low")
    tracker.save_result(result2, name="exp_high")

    comparison = tracker.compare_experiments(["exp_low", "exp_high"])

    assert len(comparison) == 2
    assert comparison[0]["best_score"] == 0.7 or comparison[1]["best_score"] == 0.7
    assert comparison[0]["best_score"] == 0.9 or comparison[1]["best_score"] == 0.9


def test_experiment_tracker_creates_directory():
    """Test that tracker creates directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "nested" / "experiments"
        tracker = ExperimentTracker(experiment_dir=str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()
