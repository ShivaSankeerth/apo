"""LLM provider integrations."""

from apo.providers.base import LLMProvider, LLMResponse
from apo.providers.anthropic import AnthropicProvider
from apo.providers.openai import OpenAIProvider

# Optional providers (may require additional dependencies)
try:
    from apo.providers.gemini import GeminiProvider
except ImportError:
    GeminiProvider = None

try:
    from apo.providers.ollama import OllamaProvider
except ImportError:
    OllamaProvider = None

try:
    from apo.providers.cohere import CohereProvider
except ImportError:
    CohereProvider = None

try:
    from apo.providers.mistral import MistralProvider
except ImportError:
    MistralProvider = None

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "CohereProvider",
    "MistralProvider",
]
