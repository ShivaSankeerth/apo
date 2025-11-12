"""Tests for Pareto frontier."""

import pytest
from apo.core.prompt import Prompt
from apo.core.evaluator import EvaluationResult
from apo.core.pareto import ParetoFrontier


def test_pareto_frontier_creation():
    """Test creating a Pareto frontier."""
    frontier = ParetoFrontier()
    assert frontier.size() == 0


def test_pareto_frontier_add_candidate():
    """Test adding candidates to frontier."""
    frontier = ParetoFrontier()

    # Add first candidate
    prompt1 = Prompt(content="Prompt 1")
    result1 = EvaluationResult(
        prompt=prompt1,
        metrics={"accuracy": 0.8, "speed": 0.6}
    )
    added = frontier.add_candidate(result1)
    assert added is True
    assert frontier.size() == 1


def test_pareto_frontier_dominance():
    """Test dominance checking."""
    frontier = ParetoFrontier(objectives=["accuracy", "speed"])

    # Add initial candidate
    prompt1 = Prompt(content="Prompt 1")
    result1 = EvaluationResult(
        prompt=prompt1,
        metrics={"accuracy": 0.7, "speed": 0.7}
    )
    frontier.add_candidate(result1)

    # Add dominated candidate (worse on both)
    prompt2 = Prompt(content="Prompt 2")
    result2 = EvaluationResult(
        prompt=prompt2,
        metrics={"accuracy": 0.6, "speed": 0.6}
    )
    added = frontier.add_candidate(result2)
    assert added is False
    assert frontier.size() == 1

    # Add dominating candidate (better on both)
    prompt3 = Prompt(content="Prompt 3")
    result3 = EvaluationResult(
        prompt=prompt3,
        metrics={"accuracy": 0.9, "speed": 0.9}
    )
    added = frontier.add_candidate(result3)
    assert added is True
    assert frontier.size() == 1  # Should replace the dominated one

    # Add non-dominated candidate (better on one, worse on other)
    prompt4 = Prompt(content="Prompt 4")
    result4 = EvaluationResult(
        prompt=prompt4,
        metrics={"accuracy": 0.95, "speed": 0.5}
    )
    added = frontier.add_candidate(result4)
    assert added is True
    assert frontier.size() == 2  # Should keep both


def test_pareto_frontier_best_for_objective():
    """Test getting best for specific objective."""
    frontier = ParetoFrontier(objectives=["accuracy", "speed"])

    prompt1 = Prompt(content="Prompt 1")
    result1 = EvaluationResult(
        prompt=prompt1,
        metrics={"accuracy": 0.9, "speed": 0.6}
    )
    frontier.add_candidate(result1)

    prompt2 = Prompt(content="Prompt 2")
    result2 = EvaluationResult(
        prompt=prompt2,
        metrics={"accuracy": 0.7, "speed": 0.9}
    )
    frontier.add_candidate(result2)

    best_accuracy = frontier.get_best_for_objective("accuracy")
    assert best_accuracy.metrics["accuracy"] == 0.9

    best_speed = frontier.get_best_for_objective("speed")
    assert best_speed.metrics["speed"] == 0.9


def test_pareto_frontier_diversity():
    """Test diversity score calculation."""
    frontier = ParetoFrontier(objectives=["accuracy", "speed"])

    # Add very similar candidates
    for i in range(3):
        prompt = Prompt(content=f"Prompt {i}")
        result = EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": 0.8 + i * 0.01, "speed": 0.8 + i * 0.01}
        )
        frontier.add_candidate(result)

    diversity1 = frontier.get_diversity_score()

    # Reset and add very different candidates
    frontier = ParetoFrontier(objectives=["accuracy", "speed"])
    candidates = [
        (0.9, 0.5),
        (0.5, 0.9),
        (0.7, 0.7)
    ]

    for i, (acc, spd) in enumerate(candidates):
        prompt = Prompt(content=f"Prompt {i}")
        result = EvaluationResult(
            prompt=prompt,
            metrics={"accuracy": acc, "speed": spd}
        )
        frontier.add_candidate(result)

    diversity2 = frontier.get_diversity_score()

    # More diverse candidates should have higher diversity score
    assert diversity2 > diversity1
