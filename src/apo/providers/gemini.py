"""Google Gemini provider."""

import time
from typing import Any, List
from apo.providers.base import LLMProvider, LLMResponse

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        **kwargs: Any
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Google AI API key
            model: Model name (gemini-pro, gemini-pro-vision, etc.)
            **kwargs: Additional parameters

        Raises:
            ImportError: If google-generativeai is not installed
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )

        super().__init__(api_key, model, **kwargs)
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using Gemini.

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse containing the completion
        """
        start_time = time.time()

        # Configure generation
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )

        # Generate response (synchronous call, but we wrap it)
        import asyncio
        response = await asyncio.to_thread(
            self.client.generate_content,
            prompt,
            generation_config=generation_config
        )

        latency_ms = (time.time() - start_time) * 1000

        # Extract content
        content = response.text if response.text else ""

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "finish_reason": response.candidates[0].finish_reason if response.candidates else None,
                "safety_ratings": [
                    {
                        "category": rating.category,
                        "probability": rating.probability
                    }
                    for rating in (response.candidates[0].safety_ratings if response.candidates else [])
                ]
            },
            tokens_used=None,  # Gemini doesn't provide token count directly
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
