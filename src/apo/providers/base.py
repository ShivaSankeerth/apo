"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    metadata: Dict[str, Any]
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str, **kwargs: Any):
        """Initialize the provider.

        Args:
            api_key: API key for the provider
            model: Model name to use
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse containing the completion
        """
        pass

    @abstractmethod
    async def complete_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> List[LLMResponse]:
        """Generate completions for multiple prompts.

        Args:
            prompts: List of prompts to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            List of LLMResponses
        """
        pass

    async def generate_prompt_variations(
        self,
        base_prompt: str,
        num_variations: int = 5,
        instructions: Optional[str] = None
    ) -> List[str]:
        """Generate variations of a prompt using the LLM.

        Args:
            base_prompt: The base prompt to create variations of
            num_variations: Number of variations to generate
            instructions: Additional instructions for variation generation

        Returns:
            List of prompt variations
        """
        meta_prompt = f"""Generate {num_variations} variations of the following prompt.
Make each variation semantically similar but with different phrasing, structure, or approach.

Original prompt:
{base_prompt}

{instructions or "Focus on improving clarity, specificity, and effectiveness."}

Generate {num_variations} variations, one per line, numbered 1-{num_variations}:"""

        response = await self.complete(meta_prompt, temperature=0.8)

        # Parse variations from response
        variations = []
        for line in response.content.split('\n'):
            line = line.strip()
            # Remove numbering if present
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove leading number and punctuation
                parts = line.split('.', 1) if '.' in line else line.split(')', 1) if ')' in line else [line]
                variation = parts[-1].strip()
                if variation:
                    variations.append(variation)

        return variations[:num_variations]
