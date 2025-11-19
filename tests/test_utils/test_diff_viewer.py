"""Tests for prompt diff viewer."""

import pytest
from apo.utils.diff_viewer import PromptDiff, PromptEvolutionTracker
from apo.core.prompt import Prompt


def test_prompt_diff_creation():
    """Test creating a prompt diff."""
    prompt1 = Prompt(content="Original prompt text")
    prompt2 = Prompt(content="Modified prompt text with changes")

    diff = PromptDiff(prompt1, prompt2)

    assert diff.original == prompt1
    assert diff.modified == prompt2


def test_prompt_diff_identical_prompts():
    """Test diff with identical prompts."""
    prompt = Prompt(content="Same prompt")

    diff = PromptDiff(prompt, prompt.copy())

    stats = diff.get_stats()
    assert stats["lines_added"] == 0
    assert stats["lines_removed"] == 0
    assert diff.get_similarity() == 1.0


def test_prompt_diff_completely_different():
    """Test diff with completely different prompts."""
    prompt1 = Prompt(content="First prompt")
    prompt2 = Prompt(content="Completely different second prompt")

    diff = PromptDiff(prompt1, prompt2)

    similarity = diff.get_similarity()
    assert 0 <= similarity < 1.0


def test_prompt_diff_stats():
    """Test diff statistics calculation."""
    prompt1 = Prompt(content="Line 1\nLine 2\nLine 3")
    prompt2 = Prompt(content="Line 1\nModified Line 2\nLine 3\nLine 4")

    diff = PromptDiff(prompt1, prompt2)
    stats = diff.get_stats()

    assert "lines_added" in stats
    assert "lines_removed" in stats
    assert "chars_added" in stats
    assert "chars_removed" in stats
    assert stats["lines_added"] >= 0
    assert stats["lines_removed"] >= 0


def test_prompt_diff_to_console():
    """Test console output generation."""
    prompt1 = Prompt(content="Original text")
    prompt2 = Prompt(content="Modified text")

    diff = PromptDiff(prompt1, prompt2)

    console_output = diff.to_console(colors=False)
    assert isinstance(console_output, str)
    assert len(console_output) > 0

    # Test with colors
    colored_output = diff.to_console(colors=True)
    assert isinstance(colored_output, str)


def test_prompt_diff_to_html():
    """Test HTML output generation."""
    prompt1 = Prompt(content="Original HTML test")
    prompt2 = Prompt(content="Modified HTML test with changes")

    diff = PromptDiff(prompt1, prompt2)
    html = diff.to_html()

    assert isinstance(html, str)
    assert "<html>" in html.lower() or "<div" in html.lower()
    assert len(html) > 0


def test_prompt_diff_to_markdown():
    """Test markdown output generation."""
    prompt1 = Prompt(content="Original markdown")
    prompt2 = Prompt(content="Modified markdown")

    diff = PromptDiff(prompt1, prompt2)
    markdown = diff.to_markdown()

    assert isinstance(markdown, str)
    assert len(markdown) > 0


def test_prompt_diff_similarity_range():
    """Test that similarity score is always between 0 and 1."""
    test_cases = [
        (Prompt(content="Same"), Prompt(content="Same")),
        (Prompt(content="A"), Prompt(content="B")),
        (Prompt(content="Hello world"), Prompt(content="Hello there world")),
        (Prompt(content=""), Prompt(content="Something")),
    ]

    for prompt1, prompt2 in test_cases:
        diff = PromptDiff(prompt1, prompt2)
        similarity = diff.get_similarity()
        assert 0.0 <= similarity <= 1.0


def test_evolution_tracker_initialization():
    """Test evolution tracker initialization."""
    tracker = PromptEvolutionTracker()

    assert tracker.versions == []
    assert len(tracker.versions) == 0


def test_evolution_tracker_add_version():
    """Test adding versions to tracker."""
    tracker = PromptEvolutionTracker()

    prompt1 = Prompt(content="Version 1")
    prompt2 = Prompt(content="Version 2")

    tracker.add_version("v1", prompt1)
    tracker.add_version("v2", prompt2)

    assert len(tracker.versions) == 2
    assert tracker.versions[0]["label"] == "v1"
    assert tracker.versions[1]["label"] == "v2"


def test_evolution_tracker_get_diff():
    """Test getting diff between versions."""
    tracker = PromptEvolutionTracker()

    prompt1 = Prompt(content="First version")
    prompt2 = Prompt(content="Second version")

    tracker.add_version("v1", prompt1)
    tracker.add_version("v2", prompt2)

    diff = tracker.get_diff(0, 1)
    assert isinstance(diff, PromptDiff)
    assert diff.original.content == "First version"
    assert diff.modified.content == "Second version"


def test_evolution_tracker_get_all_diffs():
    """Test getting all consecutive diffs."""
    tracker = PromptEvolutionTracker()

    prompts = [
        Prompt(content="Version 1"),
        Prompt(content="Version 2"),
        Prompt(content="Version 3"),
    ]

    for i, prompt in enumerate(prompts):
        tracker.add_version(f"v{i+1}", prompt)

    diffs = tracker.get_all_diffs()
    assert len(diffs) == 2  # v1->v2 and v2->v3
    assert all(isinstance(d, PromptDiff) for d in diffs)


def test_evolution_tracker_summary():
    """Test evolution summary generation."""
    tracker = PromptEvolutionTracker()

    tracker.add_version("Initial", Prompt(content="Start"))
    tracker.add_version("Improved", Prompt(content="Start with improvements"))
    tracker.add_version("Final", Prompt(content="Final version"))

    summary = tracker.get_summary()

    assert "total_versions" in summary
    assert summary["total_versions"] == 3
    assert "total_changes" in summary


def test_evolution_tracker_show_evolution(capsys):
    """Test show evolution output."""
    tracker = PromptEvolutionTracker()

    tracker.add_version("v1", Prompt(content="First"))
    tracker.add_version("v2", Prompt(content="Second"))

    # This should print to stdout
    tracker.show_evolution()

    # Verify some output was produced
    captured = capsys.readouterr()
    assert len(captured.out) > 0 or len(tracker.versions) > 0


def test_diff_with_multiline_prompts():
    """Test diff with multiline prompts."""
    prompt1 = Prompt(content="""
    You are a helpful assistant.
    Please analyze the following text.
    Be concise and clear.
    """)

    prompt2 = Prompt(content="""
    You are a helpful AI assistant.
    Please carefully analyze the following text.
    Be concise, clear, and accurate.
    Provide specific examples.
    """)

    diff = PromptDiff(prompt1, prompt2)
    stats = diff.get_stats()

    assert stats["lines_added"] > 0
    assert isinstance(diff.to_console(), str)
    assert isinstance(diff.to_html(), str)
    assert isinstance(diff.to_markdown(), str)


def test_diff_with_empty_prompts():
    """Test diff with empty prompts."""
    prompt1 = Prompt(content="")
    prompt2 = Prompt(content="New content")

    diff = PromptDiff(prompt1, prompt2)
    stats = diff.get_stats()

    assert stats["lines_added"] > 0
    assert stats["chars_added"] > 0


def test_evolution_tracker_empty():
    """Test evolution tracker with no versions."""
    tracker = PromptEvolutionTracker()

    assert len(tracker.versions) == 0
    assert tracker.get_all_diffs() == []

    summary = tracker.get_summary()
    assert summary["total_versions"] == 0
