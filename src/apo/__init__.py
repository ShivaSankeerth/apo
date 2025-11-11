"""APO: Automatic Prompt Optimization Framework."""

from apo.core.prompt import Prompt, PromptTemplate
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.strategies.genetic import GeneticOptimizer
from apo.strategies.hill_climbing import HillClimbingOptimizer
from apo.strategies.simulated_annealing import SimulatedAnnealingOptimizer
from apo.strategies.random_search import RandomSearchOptimizer
from apo.providers.anthropic import AnthropicProvider
from apo.providers.openai import OpenAIProvider

__version__ = "0.1.0"

__all__ = [
    "Prompt",
    "PromptTemplate",
    "Optimizer",
    "OptimizationResult",
    "Evaluator",
    "EvaluationResult",
    "GeneticOptimizer",
    "HillClimbingOptimizer",
    "SimulatedAnnealingOptimizer",
    "RandomSearchOptimizer",
    "AnthropicProvider",
    "OpenAIProvider",
]
