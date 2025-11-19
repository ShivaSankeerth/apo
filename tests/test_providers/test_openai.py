"""Tests for OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from apo.providers.openai import OpenAIProvider


def test_openai_provider_initialization():
    """Test OpenAI provider initialization."""
    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )

    assert provider.model == "gpt-4"
    assert provider.temperature == 0.7
    assert provider.max_tokens == 1000


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_openai_generate(mock_openai):
    """Test OpenAI text generation."""
    # Mock the API response
    mock_choice = MagicMock()
    mock_choice.message.content = "Generated response from GPT"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai.return_value = mock_client

    provider = OpenAIProvider(api_key="test-key")
    response = await provider.generate("Test prompt")

    assert response == "Generated response from GPT"
    assert mock_client.chat.completions.create.called


@pytest.mark.asyncio
@patch('openai.AsyncOpenAI')
async def test_openai_generate_with_kwargs(mock_openai):
    """Test OpenAI generation with custom parameters."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Custom response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    mock_openai.return_value = mock_client

    provider = OpenAIProvider(api_key="test-key", temperature=0.3)
    response = await provider.generate("Test prompt", max_tokens=200)

    assert response == "Custom response"
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert call_kwargs["max_tokens"] == 200


def test_openai_provider_model_options():
    """Test different OpenAI model options."""
    models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o"]

    for model in models:
        provider = OpenAIProvider(api_key="test-key", model=model)
        assert provider.model == model
