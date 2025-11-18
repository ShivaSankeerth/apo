"""Prompt diff viewer for visualizing changes between prompts."""

import difflib
from typing import List, Tuple, Optional
from enum import Enum


class DiffType(Enum):
    """Type of diff change."""
    ADDED = "added"
    REMOVED = "removed"
    UNCHANGED = "unchanged"
    MODIFIED = "modified"


class PromptDiff:
    """Visualize differences between two prompts."""

    def __init__(self, original: str, modified: str):
        """Initialize diff viewer.

        Args:
            original: Original prompt text
            modified: Modified prompt text
        """
        self.original = original
        self.modified = modified
        self._diff = None

    def get_diff(self) -> List[Tuple[DiffType, str]]:
        """Get structured diff.

        Returns:
            List of (diff_type, text) tuples
        """
        if self._diff is None:
            self._compute_diff()
        return self._diff

    def _compute_diff(self) -> None:
        """Compute the diff between prompts."""
        differ = difflib.Differ()
        diff = list(differ.compare(
            self.original.splitlines(keepends=True),
            self.modified.splitlines(keepends=True)
        ))

        self._diff = []
        for line in diff:
            if line.startswith('+ '):
                self._diff.append((DiffType.ADDED, line[2:]))
            elif line.startswith('- '):
                self._diff.append((DiffType.REMOVED, line[2:]))
            elif line.startswith('  '):
                self._diff.append((DiffType.UNCHANGED, line[2:]))
            # Skip lines starting with '? ' (diff markers)

    def to_console(self, colors: bool = True) -> str:
        """Format diff for console output.

        Args:
            colors: Whether to use ANSI color codes

        Returns:
            Formatted diff string
        """
        diff = self.get_diff()
        lines = []

        # ANSI color codes
        RED = "\033[91m" if colors else ""
        GREEN = "\033[92m" if colors else ""
        RESET = "\033[0m" if colors else ""

        for diff_type, text in diff:
            if diff_type == DiffType.ADDED:
                lines.append(f"{GREEN}+ {text}{RESET}")
            elif diff_type == DiffType.REMOVED:
                lines.append(f"{RED}- {text}{RESET}")
            else:
                lines.append(f"  {text}")

        return "".join(lines)

    def to_html(self) -> str:
        """Format diff as HTML with styling.

        Returns:
            HTML string with inline CSS
        """
        diff = self.get_diff()
        lines = []

        lines.append("""
<style>
.diff-container {
    font-family: 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.5;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 10px;
    background: #f9f9f9;
}
.diff-added {
    background-color: #d4edda;
    color: #155724;
}
.diff-removed {
    background-color: #f8d7da;
    color: #721c24;
}
.diff-unchanged {
    color: #333;
}
.diff-line {
    padding: 2px 5px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
</style>
<div class="diff-container">
""")

        for diff_type, text in diff:
            text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            if diff_type == DiffType.ADDED:
                lines.append(f'<div class="diff-line diff-added">+ {text_escaped}</div>')
            elif diff_type == DiffType.REMOVED:
                lines.append(f'<div class="diff-line diff-removed">- {text_escaped}</div>')
            else:
                lines.append(f'<div class="diff-line diff-unchanged">  {text_escaped}</div>')

        lines.append("</div>")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Format diff as markdown with code blocks.

        Returns:
            Markdown formatted diff
        """
        diff = self.get_diff()
        lines = ["```diff"]

        for diff_type, text in diff:
            if diff_type == DiffType.ADDED:
                lines.append(f"+ {text.rstrip()}")
            elif diff_type == DiffType.REMOVED:
                lines.append(f"- {text.rstrip()}")
            else:
                lines.append(f"  {text.rstrip()}")

        lines.append("```")

        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Get statistics about the changes.

        Returns:
            Dictionary with change statistics
        """
        diff = self.get_diff()

        added_lines = sum(1 for dt, _ in diff if dt == DiffType.ADDED)
        removed_lines = sum(1 for dt, _ in diff if dt == DiffType.REMOVED)
        unchanged_lines = sum(1 for dt, _ in diff if dt == DiffType.UNCHANGED)

        added_chars = sum(len(text) for dt, text in diff if dt == DiffType.ADDED)
        removed_chars = sum(len(text) for dt, text in diff if dt == DiffType.REMOVED)

        return {
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "unchanged_lines": unchanged_lines,
            "total_lines": len(diff),
            "added_chars": added_chars,
            "removed_chars": removed_chars,
            "net_change_chars": added_chars - removed_chars,
        }

    def get_similarity(self) -> float:
        """Calculate similarity ratio between prompts.

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        return difflib.SequenceMatcher(
            None,
            self.original,
            self.modified
        ).ratio()


def show_diff(
    original: str,
    modified: str,
    format: str = "console",
    colors: bool = True
) -> str:
    """Show diff between two prompts.

    Args:
        original: Original prompt
        modified: Modified prompt
        format: Output format ('console', 'html', 'markdown')
        colors: Whether to use colors (for console format)

    Returns:
        Formatted diff string
    """
    diff = PromptDiff(original, modified)

    if format == "console":
        return diff.to_console(colors=colors)
    elif format == "html":
        return diff.to_html()
    elif format == "markdown":
        return diff.to_markdown()
    else:
        raise ValueError(f"Unknown format: {format}")


def compare_prompts(
    prompts: List[Tuple[str, str]],
    show_stats: bool = True
) -> None:
    """Compare multiple prompt pairs.

    Args:
        prompts: List of (label, prompt_text) tuples
        show_stats: Whether to show statistics
    """
    if len(prompts) < 2:
        print("Need at least 2 prompts to compare")
        return

    # Compare consecutive pairs
    for i in range(len(prompts) - 1):
        label1, prompt1 = prompts[i]
        label2, prompt2 = prompts[i + 1]

        print(f"\n{'='*60}")
        print(f"Comparing: {label1} → {label2}")
        print(f"{'='*60}\n")

        diff = PromptDiff(prompt1, prompt2)
        print(diff.to_console())

        if show_stats:
            stats = diff.get_stats()
            similarity = diff.get_similarity()

            print(f"\nStatistics:")
            print(f"  Similarity: {similarity:.1%}")
            print(f"  Lines: +{stats['added_lines']} -{stats['removed_lines']}")
            print(f"  Characters: +{stats['added_chars']} -{stats['removed_chars']}")
            print(f"  Net change: {stats['net_change_chars']:+d} characters")


def highlight_changes(original: str, modified: str) -> Tuple[str, str]:
    """Highlight changes in both prompts with markers.

    Args:
        original: Original prompt
        modified: Modified prompt

    Returns:
        Tuple of (highlighted_original, highlighted_modified)
    """
    diff = PromptDiff(original, modified)
    diff_list = diff.get_diff()

    original_highlighted = []
    modified_highlighted = []

    for diff_type, text in diff_list:
        if diff_type == DiffType.REMOVED:
            original_highlighted.append(f"[-{text.rstrip()}-]")
        elif diff_type == DiffType.ADDED:
            modified_highlighted.append(f"[+{text.rstrip()}+]")
        else:
            original_highlighted.append(text.rstrip())
            modified_highlighted.append(text.rstrip())

    return ("\n".join(original_highlighted), "\n".join(modified_highlighted))


class PromptEvolutionTracker:
    """Track prompt evolution over time."""

    def __init__(self):
        """Initialize evolution tracker."""
        self.versions: List[Tuple[str, str]] = []  # (version_name, prompt_text)

    def add_version(self, version_name: str, prompt_text: str) -> None:
        """Add a new prompt version.

        Args:
            version_name: Name/label for this version
            prompt_text: The prompt text
        """
        self.versions.append((version_name, prompt_text))

    def show_evolution(self, show_all_diffs: bool = False) -> None:
        """Show prompt evolution over time.

        Args:
            show_all_diffs: If True, show all diffs. If False, show summary only.
        """
        if len(self.versions) < 2:
            print("Need at least 2 versions to show evolution")
            return

        print(f"\n{'='*60}")
        print(f"PROMPT EVOLUTION ({len(self.versions)} versions)")
        print(f"{'='*60}\n")

        if show_all_diffs:
            compare_prompts(self.versions, show_stats=True)
        else:
            # Show summary
            for i, (name, prompt) in enumerate(self.versions):
                print(f"{i+1}. {name}")
                print(f"   Length: {len(prompt)} characters")
                if i > 0:
                    prev_prompt = self.versions[i-1][1]
                    similarity = PromptDiff(prev_prompt, prompt).get_similarity()
                    print(f"   Change from previous: {(1-similarity)*100:.1f}%")

            print(f"\nOverall change:")
            first_prompt = self.versions[0][1]
            last_prompt = self.versions[-1][1]
            similarity = PromptDiff(first_prompt, last_prompt).get_similarity()
            print(f"  Similarity: {similarity:.1%}")
            print(f"  Total change: {(1-similarity)*100:.1f}%")

    def get_best_improvement(self) -> Optional[Tuple[str, str, float]]:
        """Find the version with the biggest improvement.

        Returns:
            Tuple of (from_version, to_version, change_percent) or None
        """
        if len(self.versions) < 2:
            return None

        max_change = 0.0
        best_pair = None

        for i in range(len(self.versions) - 1):
            name1, prompt1 = self.versions[i]
            name2, prompt2 = self.versions[i + 1]

            similarity = PromptDiff(prompt1, prompt2).get_similarity()
            change = (1 - similarity) * 100

            if change > max_change:
                max_change = change
                best_pair = (name1, name2, change)

        return best_pair
