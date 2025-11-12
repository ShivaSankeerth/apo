"""Pareto frontier tracking for multi-objective optimization (GEPA-inspired)."""

from typing import List, Dict, Any
from dataclasses import dataclass, field
from apo.core.evaluator import EvaluationResult


@dataclass
class ParetoFrontier:
    """Maintains a Pareto frontier of non-dominated solutions.

    Inspired by DSPy GEPA's approach of maintaining multiple complementary
    strategies instead of just tracking the single best solution.
    """

    candidates: List[EvaluationResult] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)

    def add_candidate(self, result: EvaluationResult) -> bool:
        """Add a candidate to the frontier if it's non-dominated.

        Args:
            result: Evaluation result to add

        Returns:
            True if the candidate was added, False otherwise
        """
        if not self.objectives:
            # Infer objectives from metrics
            self.objectives = list(result.metrics.keys())

        # Check if this candidate is dominated by any existing candidate
        is_dominated = False
        candidates_to_remove = []

        for i, existing in enumerate(self.candidates):
            if self._dominates(existing, result):
                is_dominated = True
                break
            elif self._dominates(result, existing):
                candidates_to_remove.append(i)

        # Remove dominated candidates
        for i in reversed(candidates_to_remove):
            self.candidates.pop(i)

        # Add if not dominated
        if not is_dominated:
            self.candidates.append(result)
            return True

        return False

    def _dominates(self, a: EvaluationResult, b: EvaluationResult) -> bool:
        """Check if candidate a dominates candidate b.

        a dominates b if a is at least as good as b on all objectives
        and strictly better on at least one objective.
        """
        at_least_as_good = True
        strictly_better = False

        for objective in self.objectives:
            a_value = a.metrics.get(objective, 0.0)
            b_value = b.metrics.get(objective, 0.0)

            if a_value < b_value:
                at_least_as_good = False
                break

            if a_value > b_value:
                strictly_better = True

        return at_least_as_good and strictly_better

    def get_best_for_objective(self, objective: str) -> EvaluationResult:
        """Get the best candidate for a specific objective."""
        if not self.candidates:
            raise ValueError("Frontier is empty")

        return max(self.candidates, key=lambda x: x.metrics.get(objective, 0.0))

    def get_random_candidate(self) -> EvaluationResult:
        """Get a random candidate from the frontier."""
        import random
        if not self.candidates:
            raise ValueError("Frontier is empty")
        return random.choice(self.candidates)

    def size(self) -> int:
        """Get the number of candidates in the frontier."""
        return len(self.candidates)

    def get_all_candidates(self) -> List[EvaluationResult]:
        """Get all candidates in the frontier."""
        return self.candidates.copy()

    def get_diversity_score(self) -> float:
        """Calculate diversity score of the frontier.

        Higher diversity means more varied solutions.
        """
        if len(self.candidates) < 2:
            return 0.0

        # Calculate average pairwise distance in objective space
        total_distance = 0.0
        pairs = 0

        for i, a in enumerate(self.candidates):
            for b in self.candidates[i + 1:]:
                distance = self._objective_distance(a, b)
                total_distance += distance
                pairs += 1

        return total_distance / pairs if pairs > 0 else 0.0

    def _objective_distance(self, a: EvaluationResult, b: EvaluationResult) -> float:
        """Calculate distance between two candidates in objective space."""
        distance_squared = 0.0

        for objective in self.objectives:
            a_value = a.metrics.get(objective, 0.0)
            b_value = b.metrics.get(objective, 0.0)
            distance_squared += (a_value - b_value) ** 2

        return distance_squared ** 0.5

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of the frontier."""
        if not self.candidates:
            return {"size": 0, "objectives": self.objectives}

        summary = {
            "size": len(self.candidates),
            "objectives": self.objectives,
            "diversity": self.get_diversity_score(),
        }

        # Add best values for each objective
        for objective in self.objectives:
            values = [c.metrics.get(objective, 0.0) for c in self.candidates]
            summary[f"best_{objective}"] = max(values)
            summary[f"mean_{objective}"] = sum(values) / len(values)

        return summary
