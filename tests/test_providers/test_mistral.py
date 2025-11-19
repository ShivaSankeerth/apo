"""Tests for Mistral AI provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from apo.providers.mistral import MistralProvider


def test_mistral_provider_initialization():
    """Test Mistral provider initialization."""
    with patch('mistralai.async_client.MistralAsyncClient'):
        provider = MistralProvider(
            api_key="test-key",
            model="mistral-small-latest",
            temperature=0.7
        )

        assert provider.model == "mistral-small-latest"
        assert provider.temperature == 0.7


@pytest.mark.asyncio
@patch('mistralai.async_client.MistralAsyncClient')
async def test_mistral_generate(mock_client_class):
    """Test Mistral text generation."""
    # Mock the API response
    mock_message = MagicMock()
    mock_message.content = "Generated response from Mistral"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)
    mock_client_class.return_value = mock_client

    provider = MistralProvider(api_key="test-key")
    response = await provider.generate("Test prompt")

    assert response == "Generated response from Mistral"
    assert mock_client.chat.called


@pytest.mark.asyncio
@patch('mistralai.async_client.MistralAsyncClient')
async def test_mistral_generate_with_kwargs(mock_client_class):
    """Test Mistral generation with custom parameters."""
    mock_message = MagicMock()
    mock_message.content = "Custom Mistral response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)
    mock_client_class.return_value = mock_client

    provider = MistralProvider(api_key="test-key", temperature=0.9)
    response = await provider.generate("Test prompt", max_tokens=1000)

    assert response == "Custom Mistral response"


def test_mistral_provider_import_error_handling():
    """Test handling of missing mistralai package."""
    with patch('apo.providers.mistral.HAS_MISTRAL', False):
        with pytest.raises(ImportError, match="mistralai"):
            MistralProvider(api_key="test-key")


def test_mistral_provider_model_options():
    """Test different Mistral model options."""
    with patch('mistralai.async_client.MistralAsyncClient'):
        models = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]

        for model in models:
            provider = MistralProvider(api_key="test-key", model=model)
            assert provider.model == model
