"""Prompt signatures for declarative prompt specification (DSPy-inspired)."""

from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional


@dataclass
class PromptSignature:
    """Declarative specification of prompt input/output behavior.

    Inspired by DSPy signatures, this allows you to specify what a prompt
    should do rather than how it should do it.
    """

    inputs: Dict[str, Type] = field(default_factory=dict)
    outputs: Dict[str, Type] = field(default_factory=dict)
    description: str = ""
    instructions: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the signature."""
        if not self.inputs:
            raise ValueError("Signature must have at least one input field")
        if not self.outputs:
            raise ValueError("Signature must have at least one output field")

    def to_prompt_template(self) -> str:
        """Convert signature to a prompt template."""
        parts = []

        if self.description:
            parts.append(self.description)

        if self.instructions:
            parts.append(f"\nInstructions: {self.instructions}")

        # Input specification
        parts.append("\nInputs:")
        for name, type_ in self.inputs.items():
            parts.append(f"- {name}: {type_.__name__}")

        # Output specification
        parts.append("\nExpected Outputs:")
        for name, type_ in self.outputs.items():
            parts.append(f"- {name}: {type_.__name__}")

        # Template for actual use
        parts.append("\n---")
        for name in self.inputs.keys():
            parts.append(f"{name}: {{{name}}}")

        return "\n".join(parts)

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that provided inputs match the signature."""
        for name, type_ in self.inputs.items():
            if name not in inputs:
                return False
            if not isinstance(inputs[name], type_):
                return False
        return True

    def validate_outputs(self, outputs: Dict[str, Any]) -> bool:
        """Validate that outputs match the signature."""
        for name, type_ in self.outputs.items():
            if name not in outputs:
                return False
            # Basic type checking (could be enhanced)
            if not isinstance(outputs[name], (type_, str)):
                return False
        return True


@dataclass
class PromptVersion:
    """Version tracking for prompts."""

    version: str
    content: str
    signature: Optional[PromptSignature] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self) -> None:
        """Set created_at if not provided."""
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()


class PromptVersionManager:
    """Manage prompt versions over time."""

    def __init__(self) -> None:
        """Initialize version manager."""
        self.versions: Dict[str, PromptVersion] = {}
        self.current_version: Optional[str] = None

    def add_version(
        self,
        version: str,
        content: str,
        signature: Optional[PromptSignature] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        parent_version: Optional[str] = None,
        **metadata: Any
    ) -> PromptVersion:
        """Add a new prompt version."""
        prompt_version = PromptVersion(
            version=version,
            content=content,
            signature=signature,
            performance_metrics=performance_metrics or {},
            metadata=metadata,
            parent_version=parent_version
        )

        self.versions[version] = prompt_version
        self.current_version = version
        return prompt_version

    def get_version(self, version: str) -> Optional[PromptVersion]:
        """Get a specific version."""
        return self.versions.get(version)

    def get_current(self) -> Optional[PromptVersion]:
        """Get the current version."""
        if self.current_version:
            return self.versions.get(self.current_version)
        return None

    def list_versions(self) -> list[str]:
        """List all version identifiers."""
        return list(self.versions.keys())

    def get_best_version(self, metric: str = "score") -> Optional[PromptVersion]:
        """Get the version with the best performance on a metric."""
        if not self.versions:
            return None

        best_version = None
        best_score = float('-inf')

        for version in self.versions.values():
            score = version.performance_metrics.get(metric, float('-inf'))
            if score > best_score:
                best_score = score
                best_version = version

        return best_version

    def get_version_history(self, version: str) -> list[PromptVersion]:
        """Get the history of versions leading to this one."""
        history = []
        current = self.get_version(version)

        while current:
            history.append(current)
            if current.parent_version:
                current = self.get_version(current.parent_version)
            else:
                break

        return list(reversed(history))
