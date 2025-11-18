"""Mistral AI provider."""

import time
from typing import Any, List
from apo.providers.base import LLMProvider, LLMResponse

try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


class MistralProvider(LLMProvider):
    """Provider for Mistral AI models."""

    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        **kwargs: Any
    ):
        """Initialize Mistral provider.

        Args:
            api_key: Mistral API key
            model: Model name (mistral-tiny, mistral-small, mistral-medium, mistral-large, etc.)
            **kwargs: Additional parameters

        Raises:
            ImportError: If mistralai is not installed
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "mistralai is not installed. "
                "Install it with: pip install mistralai"
            )

        super().__init__(api_key, model, **kwargs)
        self.client = MistralClient(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using Mistral.

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
        messages = [ChatMessage(role="user", content=prompt)]

        response = await asyncio.to_thread(
            self.client.chat,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content if response.choices else ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "id": response.id,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
            },
            tokens_used=(
                response.usage.total_tokens if hasattr(response, "usage") and response.usage else None
            ),
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
