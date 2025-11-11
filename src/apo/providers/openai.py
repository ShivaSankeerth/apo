"""OpenAI provider."""

import time
from typing import Any, List
from openai import AsyncOpenAI
from apo.providers.base import LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI's models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        **kwargs: Any
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4-turbo-preview)
            **kwargs: Additional parameters
        """
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using OpenAI.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the completion
        """
        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "id": response.id,
                "finish_reason": response.choices[0].finish_reason,
            },
            tokens_used=response.usage.total_tokens if response.usage else None,
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
