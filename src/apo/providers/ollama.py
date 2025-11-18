"""Ollama provider for local LLM models."""

import time
from typing import Any, List
from apo.providers.base import LLMProvider, LLMResponse

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class OllamaProvider(LLMProvider):
    """Provider for Ollama local models.

    Ollama allows you to run LLMs locally without API costs.
    Popular models: llama2, mistral, codellama, phi, gemma, etc.
    """

    def __init__(
        self,
        model: str = "llama2",
        host: str = "http://localhost:11434",
        **kwargs: Any
    ):
        """Initialize Ollama provider.

        Args:
            model: Model name (llama2, mistral, codellama, etc.)
            host: Ollama server host
            **kwargs: Additional parameters

        Raises:
            ImportError: If ollama package is not installed
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama is not installed. "
                "Install it with: pip install ollama"
            )

        super().__init__(api_key="", model=model, **kwargs)
        self.host = host
        self.client = ollama.Client(host=host)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using Ollama.

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
            model=self.model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.get("response", "")

        return LLMResponse(
            content=content,
            model=self.model,
            metadata={
                "context": response.get("context"),
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
            },
            tokens_used=response.get("eval_count"),
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

    def list_models(self) -> List[str]:
        """List available models on Ollama server.

        Returns:
            List of model names
        """
        models = self.client.list()
        return [model["name"] for model in models.get("models", [])]

    def pull_model(self, model: str) -> None:
        """Pull a model from Ollama library.

        Args:
            model: Model name to pull (e.g., "llama2", "mistral")
        """
        self.client.pull(model)
