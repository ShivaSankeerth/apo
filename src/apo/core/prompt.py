"""Prompt representation and templating."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from jinja2 import Template
import hashlib


@dataclass
class Prompt:
    """Represents a prompt with its content and metadata."""

    content: str
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None

    def __hash__(self) -> int:
        """Generate a hash of the prompt content."""
        return int(hashlib.md5(self.content.encode()).hexdigest(), 16)

    def __eq__(self, other: object) -> bool:
        """Check equality based on content."""
        if not isinstance(other, Prompt):
            return False
        return self.content == other.content

    def render(self, **kwargs: Any) -> str:
        """Render the prompt with given variables."""
        template = Template(self.content)
        all_vars = {**self.variables, **kwargs}
        return template.render(**all_vars)

    def copy(self) -> "Prompt":
        """Create a copy of this prompt."""
        return Prompt(
            content=self.content,
            variables=self.variables.copy(),
            metadata=self.metadata.copy(),
            score=self.score,
        )


@dataclass
class PromptTemplate:
    """Template for generating prompt variations."""

    base_template: str
    components: List[str] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def build(self, **kwargs: Any) -> Prompt:
        """Build a prompt from the template."""
        parts = [self.base_template]

        if self.instructions:
            parts.append("\nInstructions:")
            parts.extend([f"- {inst}" for inst in self.instructions])

        if self.examples:
            parts.append("\nExamples:")
            parts.extend(self.examples)

        content = "\n".join(parts)
        return Prompt(content=content, variables=kwargs)

    def add_instruction(self, instruction: str) -> None:
        """Add an instruction to the template."""
        self.instructions.append(instruction)

    def add_example(self, example: str) -> None:
        """Add an example to the template."""
        self.examples.append(example)

    def remove_instruction(self, index: int) -> None:
        """Remove an instruction by index."""
        if 0 <= index < len(self.instructions):
            self.instructions.pop(index)

    def remove_example(self, index: int) -> None:
        """Remove an example by index."""
        if 0 <= index < len(self.examples):
            self.examples.pop(index)
