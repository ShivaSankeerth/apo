"""Tests for Ollama provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from apo.providers.ollama import OllamaProvider


def test_ollama_provider_initialization():
    """Test Ollama provider initialization."""
    provider = OllamaProvider(
        model="llama2",
        host="http://localhost:11434",
        temperature=0.7
    )

    assert provider.model == "llama2"
    assert provider.host == "http://localhost:11434"
    assert provider.temperature == 0.7


@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_ollama_generate(mock_session_class):
    """Test Ollama text generation."""
    # Mock the API response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "response": "Generated response from Ollama"
    })

    mock_session = AsyncMock()
    mock_session.post = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session_class.return_value = mock_session

    provider = OllamaProvider(model="llama2")
    response = await provider.generate("Test prompt")

    assert response == "Generated response from Ollama"
    assert mock_session.post.called


@pytest.mark.asyncio
@patch('aiohttp.ClientSession')
async def test_ollama_list_models(mock_session_class):
    """Test listing available Ollama models."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "models": [
            {"name": "llama2"},
            {"name": "mistral"},
            {"name": "codellama"}
        ]
    })

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session_class.return_value = mock_session

    provider = OllamaProvider(model="llama2")
    models = await provider.list_models()

    assert len(models) == 3
    assert "llama2" in models


def test_ollama_provider_import_error_handling():
    """Test handling of missing ollama package."""
    with patch('apo.providers.ollama.HAS_OLLAMA', False):
        with pytest.raises(ImportError, match="ollama"):
            OllamaProvider(model="llama2")


def test_ollama_provider_custom_host():
    """Test Ollama provider with custom host."""
    provider = OllamaProvider(
        model="mistral",
        host="http://192.168.1.100:11434"
    )

    assert provider.host == "http://192.168.1.100:11434"
