"""Tests for Google Gemini provider."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from apo.providers.gemini import GeminiProvider


def test_gemini_provider_initialization():
    """Test Gemini provider initialization."""
    with patch('google.generativeai.configure'):
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-pro",
            temperature=0.7
        )

        assert provider.model == "gemini-pro"
        assert provider.temperature == 0.7


@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
@patch('google.generativeai.configure')
async def test_gemini_generate(mock_configure, mock_model_class):
    """Test Gemini text generation."""
    # Mock the API response
    mock_response = MagicMock()
    mock_response.text = "Generated response from Gemini"

    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    mock_model_class.return_value = mock_model

    provider = GeminiProvider(api_key="test-key")
    response = await provider.generate("Test prompt")

    assert response == "Generated response from Gemini"
    assert mock_model.generate_content_async.called


@pytest.mark.asyncio
@patch('google.generativeai.GenerativeModel')
@patch('google.generativeai.configure')
async def test_gemini_generate_with_config(mock_configure, mock_model_class):
    """Test Gemini generation with custom config."""
    mock_response = MagicMock()
    mock_response.text = "Custom Gemini response"

    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    mock_model_class.return_value = mock_model

    provider = GeminiProvider(api_key="test-key", temperature=0.9)
    response = await provider.generate("Test prompt")

    assert response == "Custom Gemini response"


def test_gemini_provider_import_error_handling():
    """Test handling of missing google-generativeai package."""
    with patch('apo.providers.gemini.HAS_GEMINI', False):
        with pytest.raises(ImportError, match="google-generativeai"):
            GeminiProvider(api_key="test-key")
