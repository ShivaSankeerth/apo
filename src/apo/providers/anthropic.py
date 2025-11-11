"""Anthropic Claude provider."""

import time
from typing import Any, List
from anthropic import AsyncAnthropic
from apo.providers.base import LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs: Any
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-5-sonnet-20241022)
            **kwargs: Additional parameters
        """
        super().__init__(api_key, model, **kwargs)
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using Claude.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the completion
        """
        start_time = time.time()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "id": response.id,
                "stop_reason": response.stop_reason,
            },
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
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
