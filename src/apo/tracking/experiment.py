"""Experiment tracking for optimization runs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import asdict
from apo.core.optimizer import OptimizationResult


class ExperimentTracker:
    """Track and save optimization experiments."""

    def __init__(self, experiment_dir: str = "experiments"):
        """Initialize experiment tracker.

        Args:
            experiment_dir: Directory to save experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)

    def save_result(
        self,
        result: OptimizationResult,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Save an optimization result.

        Args:
            result: The optimization result to save
            name: Optional name for the experiment
            metadata: Optional additional metadata

        Returns:
            Path to the saved file
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"experiment_{timestamp}"

        # Prepare data
        data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "best_prompt": result.best_prompt.content,
            "best_score": result.best_score,
            "duration_seconds": result.duration,
            "total_evaluations": result.total_evaluations,
            "metadata": {
                **result.metadata,
                **(metadata or {})
            },
            "history": [
                {
                    "prompt": eval_result.prompt.content,
                    "score": eval_result.score,
                    "metrics": eval_result.metrics,
                }
                for eval_result in result.history
            ],
        }

        # Save to file
        file_path = self.experiment_dir / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        return file_path

    def load_result(self, name: str) -> Dict[str, Any]:
        """Load a saved experiment result.

        Args:
            name: Name of the experiment

        Returns:
            Dictionary containing the experiment data
        """
        file_path = self.experiment_dir / f"{name}.json"
        with open(file_path, 'r') as f:
            return json.load(f)

    def list_experiments(self) -> List[str]:
        """List all saved experiments.

        Returns:
            List of experiment names
        """
        return [f.stem for f in self.experiment_dir.glob("*.json")]

    def compare_experiments(self, names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments.

        Args:
            names: List of experiment names to compare

        Returns:
            Comparison data
        """
        experiments = [self.load_result(name) for name in names]

        comparison = {
            "experiments": names,
            "best_scores": [exp["best_score"] for exp in experiments],
            "total_evaluations": [exp["total_evaluations"] for exp in experiments],
            "durations": [exp["duration_seconds"] for exp in experiments],
        }

        # Find overall best
        best_idx = max(range(len(experiments)), key=lambda i: experiments[i]["best_score"])
        comparison["overall_best"] = {
            "name": names[best_idx],
            "score": experiments[best_idx]["best_score"],
            "prompt": experiments[best_idx]["best_prompt"],
        }

        return comparison

    def get_summary_statistics(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for an experiment.

        Args:
            name: Name of the experiment

        Returns:
            Summary statistics
        """
        data = self.load_result(name)
        scores = [h["score"] for h in data["history"]]

        return {
            "name": name,
            "best_score": data["best_score"],
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "total_evaluations": len(scores),
            "duration_seconds": data["duration_seconds"],
        }
