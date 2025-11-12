"""APO: Automatic Prompt Optimization Framework."""

from apo.core.prompt import Prompt, PromptTemplate
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.core.signature import PromptSignature, PromptVersion, PromptVersionManager
from apo.core.pareto import ParetoFrontier
from apo.strategies.genetic import GeneticOptimizer
from apo.strategies.hill_climbing import HillClimbingOptimizer
from apo.strategies.simulated_annealing import SimulatedAnnealingOptimizer
from apo.strategies.random_search import RandomSearchOptimizer
from apo.strategies.meta_prompt import MetaPromptOptimizer
from apo.strategies.reflection import ReflectionOptimizer
from apo.strategies.few_shot import FewShotOptimizer
from apo.strategies.bootstrap import BootstrapOptimizer
from apo.providers.anthropic import AnthropicProvider
from apo.providers.openai import OpenAIProvider

__version__ = "0.2.0"

__all__ = [
    # Core
    "Prompt",
    "PromptTemplate",
    "Optimizer",
    "OptimizationResult",
    "Evaluator",
    "EvaluationResult",
    "PromptSignature",
    "PromptVersion",
    "PromptVersionManager",
    "ParetoFrontier",
    # Classic Strategies
    "GeneticOptimizer",
    "HillClimbingOptimizer",
    "SimulatedAnnealingOptimizer",
    "RandomSearchOptimizer",
    # Advanced Strategies (inspired by Arize & DSPy)
    "MetaPromptOptimizer",
    "ReflectionOptimizer",
    "FewShotOptimizer",
    "BootstrapOptimizer",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
]
