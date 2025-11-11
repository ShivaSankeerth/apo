"""Tests for prompt module."""

import pytest
from apo.core.prompt import Prompt, PromptTemplate


def test_prompt_creation():
    """Test basic prompt creation."""
    prompt = Prompt(content="Test prompt")
    assert prompt.content == "Test prompt"
    assert prompt.variables == {}
    assert prompt.metadata == {}
    assert prompt.score is None


def test_prompt_render():
    """Test prompt rendering with variables."""
    prompt = Prompt(
        content="Hello {name}, you are {age} years old",
        variables={"name": "Alice"}
    )
    rendered = prompt.render(age=30)
    assert "Alice" in rendered
    assert "30" in rendered


def test_prompt_copy():
    """Test prompt copying."""
    original = Prompt(content="Test", score=0.8)
    copy = original.copy()

    assert copy.content == original.content
    assert copy.score == original.score
    assert copy is not original


def test_prompt_equality():
    """Test prompt equality."""
    prompt1 = Prompt(content="Test")
    prompt2 = Prompt(content="Test")
    prompt3 = Prompt(content="Different")

    assert prompt1 == prompt2
    assert prompt1 != prompt3


def test_prompt_template():
    """Test prompt template."""
    template = PromptTemplate(
        base_template="Classify the sentiment",
        instructions=["Be concise", "Use only positive/negative/neutral"],
        examples=["Example 1", "Example 2"]
    )

    prompt = template.build()
    assert "Classify the sentiment" in prompt.content
    assert "Be concise" in prompt.content
    assert "Example 1" in prompt.content


def test_prompt_template_modification():
    """Test modifying prompt template."""
    template = PromptTemplate(base_template="Base")

    template.add_instruction("New instruction")
    assert "New instruction" in template.instructions

    template.add_example("New example")
    assert "New example" in template.examples

    template.remove_instruction(0)
    assert "New instruction" not in template.instructions
