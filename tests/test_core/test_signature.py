"""Tests for prompt signatures."""

import pytest
from apo.core.signature import PromptSignature, PromptVersion, PromptVersionManager


def test_prompt_signature_creation():
    """Test creating a prompt signature."""
    sig = PromptSignature(
        inputs={"text": str},
        outputs={"sentiment": str},
        description="Classify sentiment"
    )
    assert sig.inputs == {"text": str}
    assert sig.outputs == {"sentiment": str}
    assert sig.description == "Classify sentiment"


def test_prompt_signature_validation():
    """Test signature requires inputs and outputs."""
    with pytest.raises(ValueError):
        PromptSignature(outputs={"result": str})

    with pytest.raises(ValueError):
        PromptSignature(inputs={"text": str})


def test_prompt_signature_to_template():
    """Test converting signature to template."""
    sig = PromptSignature(
        inputs={"text": str, "context": str},
        outputs={"answer": str},
        description="Answer questions"
    )

    template = sig.to_prompt_template()
    assert "Answer questions" in template
    assert "text" in template
    assert "context" in template
    assert "answer" in template


def test_prompt_version_manager():
    """Test version management."""
    manager = PromptVersionManager()

    # Add versions
    v1 = manager.add_version(
        version="v1",
        content="Classify: {text}",
        performance_metrics={"accuracy": 0.7}
    )

    v2 = manager.add_version(
        version="v2",
        content="Classify sentiment: {text}",
        performance_metrics={"accuracy": 0.8},
        parent_version="v1"
    )

    assert manager.current_version == "v2"
    assert manager.get_version("v1") == v1
    assert manager.get_version("v2") == v2


def test_version_manager_best():
    """Test getting best version."""
    manager = PromptVersionManager()

    manager.add_version("v1", "prompt1", performance_metrics={"score": 0.5})
    manager.add_version("v2", "prompt2", performance_metrics={"score": 0.9})
    manager.add_version("v3", "prompt3", performance_metrics={"score": 0.7})

    best = manager.get_best_version(metric="score")
    assert best.version == "v2"
    assert best.performance_metrics["score"] == 0.9


def test_version_history():
    """Test getting version history."""
    manager = PromptVersionManager()

    manager.add_version("v1", "prompt1")
    manager.add_version("v2", "prompt2", parent_version="v1")
    manager.add_version("v3", "prompt3", parent_version="v2")

    history = manager.get_version_history("v3")
    assert len(history) == 3
    assert history[0].version == "v1"
    assert history[1].version == "v2"
    assert history[2].version == "v3"
