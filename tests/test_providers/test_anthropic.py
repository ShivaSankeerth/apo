"""Tests for Anthropic provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from apo.providers.anthropic import AnthropicProvider


def test_anthropic_provider_initialization():
    """Test Anthropic provider initialization."""
    provider = AnthropicProvider(
        api_key="test-key",
        model="claude-3-5-sonnet-20241022",
        temperature=0.7,
        max_tokens=1000
    )

    assert provider.model == "claude-3-5-sonnet-20241022"
    assert provider.temperature == 0.7
    assert provider.max_tokens == 1000


@pytest.mark.asyncio
@patch('anthropic.AsyncAnthropic')
async def test_anthropic_generate(mock_anthropic):
    """Test Anthropic text generation."""
    # Mock the API response
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Generated response from Claude")]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    mock_anthropic.return_value = mock_client

    provider = AnthropicProvider(api_key="test-key")
    response = await provider.generate("Test prompt")

    assert response == "Generated response from Claude"
    assert mock_client.messages.create.called


@pytest.mark.asyncio
@patch('anthropic.AsyncAnthropic')
async def test_anthropic_generate_with_kwargs(mock_anthropic):
    """Test Anthropic generation with custom parameters."""
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Custom response")]

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)
    mock_anthropic.return_value = mock_client

    provider = AnthropicProvider(api_key="test-key", temperature=0.5)
    response = await provider.generate("Test prompt", max_tokens=500)

    assert response == "Custom response"
    call_kwargs = mock_client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 500


def test_anthropic_provider_model_options():
    """Test different Anthropic model options."""
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]

    for model in models:
        provider = AnthropicProvider(api_key="test-key", model=model)
        assert provider.model == model
