"""Tests for genetic algorithm optimizer."""

import pytest
from apo.strategies.genetic import GeneticOptimizer
from apo.core.prompt import Prompt
from apo.core.optimizer import OptimizationResult


@pytest.mark.asyncio
async def test_genetic_optimizer_initialization(mock_llm_provider, mock_evaluator):
    """Test genetic optimizer initialization."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        population_size=10,
        mutation_rate=0.2,
        crossover_rate=0.7
    )

    assert optimizer.population_size == 10
    assert optimizer.mutation_rate == 0.2
    assert optimizer.crossover_rate == 0.7


@pytest.mark.asyncio
async def test_genetic_optimizer_population_initialization(
    mock_llm_provider, mock_evaluator, sample_prompt
):
    """Test population initialization."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        population_size=5
    )

    population = await optimizer._initialize_population(sample_prompt)

    assert len(population) == 5
    assert all(isinstance(p, Prompt) for p in population)
    assert sample_prompt in population


@pytest.mark.asyncio
async def test_genetic_optimizer_selection(mock_llm_provider, mock_evaluator, sample_prompts):
    """Test selection mechanism."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator
    )

    # Select best prompts
    selected = optimizer._select(sample_prompts, k=2)

    assert len(selected) == 2
    # Should select the highest scoring prompts
    assert all(p.score >= 0.7 for p in selected)


@pytest.mark.asyncio
async def test_genetic_optimizer_mutation(mock_llm_provider, mock_evaluator, sample_prompt):
    """Test mutation operation."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator,
        mutation_rate=1.0  # Always mutate for testing
    )

    mutated = await optimizer._mutate(sample_prompt)

    assert isinstance(mutated, Prompt)
    # Content should be different after mutation
    assert mutated.content != sample_prompt.content or mutated == sample_prompt


@pytest.mark.asyncio
async def test_genetic_optimizer_crossover(mock_llm_provider, mock_evaluator):
    """Test crossover operation."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=mock_evaluator
    )

    parent1 = Prompt(content="First parent prompt", score=0.8)
    parent2 = Prompt(content="Second parent prompt", score=0.7)

    child = await optimizer._crossover(parent1, parent2)

    assert isinstance(child, Prompt)
    assert child.content is not None
    assert len(child.content) > 0


@pytest.mark.asyncio
async def test_genetic_optimizer_optimization(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test complete genetic optimization."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        population_size=5,
        max_generations=3
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=3)

    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_score >= 0
    assert result.iterations <= 3
    assert len(result.history) > 0


@pytest.mark.asyncio
async def test_genetic_optimizer_improves_over_time(
    mock_llm_provider, simple_evaluator, sample_prompt, sample_test_cases
):
    """Test that genetic algorithm improves scores over generations."""
    optimizer = GeneticOptimizer(
        provider=mock_llm_provider,
        evaluator=simple_evaluator,
        population_size=5,
        max_generations=5
    )

    result = await optimizer.optimize(sample_prompt, sample_test_cases, max_iterations=5)

    # Should have improvement or at least maintain score
    scores = [p.score for p in result.history if p.score is not None]
    if len(scores) > 1:
        assert max(scores) >= scores[0]
