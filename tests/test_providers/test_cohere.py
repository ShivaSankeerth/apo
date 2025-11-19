"""Tests for Cohere provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from apo.providers.cohere import CohereProvider


def test_cohere_provider_initialization():
    """Test Cohere provider initialization."""
    with patch('cohere.AsyncClient'):
        provider = CohereProvider(
            api_key="test-key",
            model="command",
            temperature=0.7
        )

        assert provider.model == "command"
        assert provider.temperature == 0.7


@pytest.mark.asyncio
@patch('cohere.AsyncClient')
async def test_cohere_generate(mock_client_class):
    """Test Cohere text generation."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.text = "Generated response from Cohere"

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=mock_response)
    mock_client_class.return_value = mock_client

    provider = CohereProvider(api_key="test-key")
    response = await provider.generate("Test prompt")

    assert response == "Generated response from Cohere"
    assert mock_client.generate.called


@pytest.mark.asyncio
@patch('cohere.AsyncClient')
async def test_cohere_generate_with_kwargs(mock_client_class):
    """Test Cohere generation with custom parameters."""
    mock_response = MagicMock()
    mock_response.text = "Custom Cohere response"

    mock_client = AsyncMock()
    mock_client.generate = AsyncMock(return_value=mock_response)
    mock_client_class.return_value = mock_client

    provider = CohereProvider(api_key="test-key", temperature=0.5)
    response = await provider.generate("Test prompt", max_tokens=500)

    assert response == "Custom Cohere response"


def test_cohere_provider_import_error_handling():
    """Test handling of missing cohere package."""
    with patch('apo.providers.cohere.HAS_COHERE', False):
        with pytest.raises(ImportError, match="cohere"):
            CohereProvider(api_key="test-key")


def test_cohere_provider_model_options():
    """Test different Cohere model options."""
    with patch('cohere.AsyncClient'):
        models = ["command", "command-light", "command-nightly"]

        for model in models:
            provider = CohereProvider(api_key="test-key", model=model)
            assert provider.model == model
