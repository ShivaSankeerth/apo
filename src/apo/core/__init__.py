"""Core abstractions for the APO framework."""

from apo.core.prompt import Prompt, PromptTemplate
from apo.core.optimizer import Optimizer, OptimizationResult
from apo.core.evaluator import Evaluator, EvaluationResult
from apo.core.signature import PromptSignature, PromptVersion, PromptVersionManager
from apo.core.pareto import ParetoFrontier

__all__ = [
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
]
