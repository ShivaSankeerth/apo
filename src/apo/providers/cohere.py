"""Cohere provider."""

import time
from typing import Any, List
from apo.providers.base import LLMProvider, LLMResponse

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class CohereProvider(LLMProvider):
    """Provider for Cohere's models."""

    def __init__(
        self,
        api_key: str,
        model: str = "command",
        **kwargs: Any
    ):
        """Initialize Cohere provider.

        Args:
            api_key: Cohere API key
            model: Model name (command, command-light, command-nightly, etc.)
            **kwargs: Additional parameters

        Raises:
            ImportError: If cohere is not installed
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere is not installed. "
                "Install it with: pip install cohere"
            )

        super().__init__(api_key, model, **kwargs)
        self.client = cohere.Client(api_key)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using Cohere.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the completion
        """
        start_time = time.time()

        # Generate response (synchronous, wrap in async)
        import asyncio
        response = await asyncio.to_thread(
            self.client.generate,
            prompt=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.generations[0].text if response.generations else ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "id": response.id if hasattr(response, "id") else None,
                "finish_reason": response.generations[0].finish_reason if response.generations else None,
            },
            tokens_used=None,  # Cohere provides token info differently
            latency_ms=latency_ms,
        )

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
            **kwargs: Additional parameters

        Returns:
            List of LLMResponses
        """
        import asyncio
        tasks = [
            self.complete(prompt, temperature, max_tokens, **kwargs)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)
